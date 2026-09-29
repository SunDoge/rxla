//! Qwen-Image-2.1 W8 text-to-image generation from cached Qwen3-VL embeddings.

use clap::Parser;
use image::{ImageBuffer, Rgba};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::qwen_image_2_1::{
    QwenImage21Config, QwenImage21FlowScheduler, qwen_image_2_1_euler_step,
    qwen_image_2_1_vae_decoder,
};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use serde::Deserialize;
use std::{fs, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Generate an RGBA image with Qwen-Image-2.1 W8")]
struct Args {
    /// Trusted CUDA PJRT plugin.
    plugin: PathBuf,
    /// RXLA grouped-W8 transformer safetensors.
    transformer: PathBuf,
    /// Official VAE safetensors.
    vae: PathBuf,
    /// Official VAE config.json containing latent mean/std.
    vae_config: PathBuf,
    /// Safetensors containing `prompt_embeds` with shape [1, tokens, 4096].
    conditioning: PathBuf,
    /// Output PNG.
    output: PathBuf,
    #[arg(long, default_value_t = 512)]
    width: u32,
    #[arg(long, default_value_t = 512)]
    height: u32,
    #[arg(long, default_value_t = 40)]
    steps: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Deserialize)]
struct VaeConfig {
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

fn denoise(
    client: &Client,
    checkpoint_path: &PathBuf,
    conditioning_path: &PathBuf,
    latent_height: i64,
    latent_width: i64,
    steps: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    let mut conditioning = SafeTensors::open(conditioning_path)?;
    let prompt = conditioning.read_f32("prompt_embeds")?;
    if prompt.shape.len() != 3 || prompt.shape[0] != 1 || prompt.shape[2] != 4096 {
        return Err("prompt_embeds must have shape [1, tokens, 4096]".into());
    }
    let context_tokens = prompt.shape[1];
    let image_tokens = latent_height * latent_width;
    let config = QwenImage21Config::official_w8(latent_height, latent_width);
    let model = Model::new(|cx: Cx, sample, timestep, step_size, context| {
        qwen_image_2_1_euler_step(cx, &sample, &timestep, &step_size, &context, &config)
    })
    .inputs((
        ModelInput::new([1, image_tokens, 64]),
        ModelInput::new([1]),
        ModelInput::new([1]),
        ModelInput::new([1, context_tokens, 4096]),
    ))
    .trace()?;
    let mut checkpoint = SafeTensors::open(checkpoint_path)?;
    checkpoint.require_metadata(&[
        ("format", "rxla-qwen-image-2.1-w8-v1"),
        ("group_size", "64"),
    ])?;
    let parameters = checkpoint.load_parameter_schema(client, model.schema())?;
    let compiled = model.compile(
        &mut Compiler::new(client.clone(), CacheLimits::default())
            .with_compute_dtype(rxla_core::DType::BF16)
            .with_cudnn_attention(true),
    )?;
    let runner = compiled.bind_parameters(parameters.bindings())?;
    let context = client.buffer(&prompt.shape, &prompt.values)?;
    let mut random = ChaCha8Rng::seed_from_u64(seed);
    let initial = (0..image_tokens * 64)
        .map(|_| StandardNormal.sample(&mut random))
        .collect::<Vec<f32>>();
    let mut sample = client.buffer(&[1, image_tokens, 64], &initial)?;
    let schedule = QwenImage21FlowScheduler::new(steps, image_tokens as usize)?;
    for index in 0..schedule.len() {
        let step = schedule.step(index)?;
        let timestep = client.buffer(&[1], &[step.timestep])?;
        let step_size = client.buffer(&[1], &[step.step_size])?;
        sample = runner.run((&sample, &timestep, &step_size, &context))?;
        eprintln!("step={}/{} sigma={:.6}", index + 1, steps, step.timestep);
    }
    Ok(sample.to_vec::<f32>()?)
}

fn decode(
    client: &Client,
    checkpoint_path: &PathBuf,
    latent: &[f32],
    latent_height: i64,
    latent_width: i64,
) -> Result<Vec<f32>> {
    let image_tokens = latent_height * latent_width;
    let model = Model::new(|cx: Cx, latent| {
        qwen_image_2_1_vae_decoder(cx, &latent, latent_height, latent_width)
    })
    .inputs(ModelInput::new([1, image_tokens, 64]))
    .trace()?;
    let mut checkpoint = SafeTensors::open(checkpoint_path)?;
    let parameters = checkpoint.load_parameter_schema(client, model.schema())?;
    let compiled = model.compile(&mut Compiler::new(client.clone(), CacheLimits::default()))?;
    let runner = compiled.bind_parameters(parameters.bindings())?;
    let latent = client.buffer(&[1, image_tokens, 64], latent)?;
    let image: Buffer = runner.run(&latent)?;
    Ok(image.to_vec::<f32>()?)
}

fn run(args: Args) -> Result<()> {
    if args.width == 0
        || args.height == 0
        || !args.width.is_multiple_of(16)
        || !args.height.is_multiple_of(16)
    {
        return Err("width and height must be positive multiples of 16".into());
    }
    let client = unsafe { Client::load(&args.plugin) }?;
    let latent_height = i64::from(args.height / 16);
    let latent_width = i64::from(args.width / 16);
    let mut latent = denoise(
        &client,
        &args.transformer,
        &args.conditioning,
        latent_height,
        latent_width,
        args.steps,
        args.seed,
    )?;
    let vae_config: VaeConfig = serde_json::from_slice(&fs::read(&args.vae_config)?)?;
    if vae_config.latents_mean.len() != 64 || vae_config.latents_std.len() != 64 {
        return Err("VAE config must contain 64 latent means and standard deviations".into());
    }
    for token in latent.as_chunks_mut::<64>().0 {
        for (channel, value) in token.iter_mut().enumerate() {
            *value = *value * vae_config.latents_std[channel] + vae_config.latents_mean[channel];
        }
    }
    let values = decode(&client, &args.vae, &latent, latent_height, latent_width)?;
    let pixels = values
        .into_iter()
        .map(|value| ((value + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(args.width, args.height, pixels)
        .ok_or("VAE output does not match requested dimensions")?;
    image.save(&args.output)?;
    println!("saved {}", args.output.display());
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
