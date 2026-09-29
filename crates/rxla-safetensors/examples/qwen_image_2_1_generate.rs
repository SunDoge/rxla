//! Qwen-Image-2.1 W8 text-to-image generation from cached Qwen3-VL embeddings.

use clap::Parser;
use image::{ImageBuffer, Rgba};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rxla_core::{Buffer, CacheLimits, Client, ClientOptions, Compiler, Memory};
use rxla_models::qwen_image_2_1::{
    QwenImage21Config, QwenImage21FlowScheduler, qwen_image_2_1_denoise, qwen_image_2_1_euler_step,
    qwen_image_2_1_vae_decoder, qwen_image_2_1_vae_decoder_raw,
};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::{SafeTensors, WeightResidencyPool};
use serde::Deserialize;
use std::{fs, path::PathBuf, time::Instant};

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
    /// VAE tile side in output pixels. Larger images are decoded as overlapping tiles.
    #[arg(long, default_value_t = 1024)]
    vae_tile: u32,
    /// VAE tile overlap in output pixels.
    #[arg(long, default_value_t = 128)]
    vae_overlap: u32,
    /// PJRT allocator fraction; lower this when sharing the GPU.
    #[arg(long)]
    memory_fraction: Option<f32>,
    /// Reserve the complete allocator pool at client creation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    preallocate: bool,
    /// Put all denoising steps in one StableHLO while loop. The asynchronous
    /// host enqueue path remains faster on the currently tested CUDA plugin.
    #[arg(long)]
    device_loop: bool,
    /// Persistent serialized PJRT executable cache.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Retain transformer weights in PJRT pinned-host memory and prefetch them
    /// only for denoising. Intended for multi-component serving under a GPU
    /// memory budget.
    #[arg(long)]
    offload_transformer: bool,
    /// Device allocator space kept outside the weight pool for activations and
    /// executable temporaries.
    #[arg(long, default_value_t = 2048)]
    offload_headroom_mib: usize,
}

#[derive(Deserialize)]
struct VaeConfig {
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

fn compiler(client: &Client, cache_dir: Option<&PathBuf>, _namespace: &str) -> Result<Compiler> {
    let compiler = Compiler::new(client.clone(), CacheLimits::default());
    #[cfg(feature = "disk-cache")]
    let compiler = if let Some(directory) = cache_dir {
        let cache = unsafe {
            rxla_core::DiskCache::new_for_client(
                directory,
                _namespace,
                client,
                4 * 1024 * 1024 * 1024,
            )?
        };
        compiler.with_disk_cache(cache)
    } else {
        compiler
    };
    #[cfg(not(feature = "disk-cache"))]
    if cache_dir.is_some() {
        return Err("--cache-dir requires rxla-safetensors/disk-cache".into());
    }
    Ok(compiler)
}

struct DenoiseOptions<'a> {
    checkpoint_path: &'a PathBuf,
    conditioning_path: &'a PathBuf,
    latent_height: i64,
    latent_width: i64,
    steps: usize,
    seed: u64,
    device_loop: bool,
    cache_dir: Option<&'a PathBuf>,
    offload_transformer: bool,
    offload_headroom_bytes: usize,
}

fn cuda_offload_memories(client: &Client) -> Result<(Memory, Memory)> {
    let memories = client.memories()?;
    let host = memories
        .iter()
        .find(|memory| {
            memory
                .info()
                .is_ok_and(|info| info.kind.contains("pinned_host"))
        })
        .cloned()
        .ok_or("PJRT client does not expose pinned-host memory")?;
    let device = client.default_memory()?;
    Ok((host, device))
}

fn denoise(client: &Client, options: &DenoiseOptions<'_>) -> Result<Vec<f32>> {
    let mut conditioning = SafeTensors::open(options.conditioning_path)?;
    let prompt = conditioning.read_f32("prompt_embeds")?;
    if prompt.shape.len() != 3 || prompt.shape[0] != 1 || prompt.shape[2] != 4096 {
        return Err("prompt_embeds must have shape [1, tokens, 4096]".into());
    }
    let context_tokens = prompt.shape[1];
    let image_tokens = options.latent_height * options.latent_width;
    let config = QwenImage21Config::official_w8(options.latent_height, options.latent_width);
    let schedule = QwenImage21FlowScheduler::new(options.steps, image_tokens as usize)?;
    let trace_start = Instant::now();
    let model = if options.device_loop {
        Model::new(|cx: Cx, sample, context| {
            qwen_image_2_1_denoise(cx, &sample, &context, &schedule, &config)
        })
        .inputs((
            ModelInput::new([1, image_tokens, 64]),
            ModelInput::new([1, context_tokens, 4096]),
        ))
        .trace()?
    } else {
        Model::new(|cx: Cx, sample, timestep, step_size, context| {
            qwen_image_2_1_euler_step(cx, &sample, &timestep, &step_size, &context, &config)
        })
        .inputs((
            ModelInput::new([1, image_tokens, 64]),
            ModelInput::new([1]),
            ModelInput::new([1]),
            ModelInput::new([1, context_tokens, 4096]),
        ))
        .trace()?
    };
    let trace_time = trace_start.elapsed();
    let load_start = Instant::now();
    let mut checkpoint = SafeTensors::open(options.checkpoint_path)?;
    checkpoint.require_metadata(&[
        ("format", "rxla-qwen-image-2.1-w8-v1"),
        ("group_size", "64"),
    ])?;
    let (offloaded, resident) = if options.offload_transformer {
        let (host, _) = cuda_offload_memories(client)?;
        let buffers = checkpoint.load_parameter_schema(&host, model.schema())?;
        (Some(buffers.offload_to(&host)?), None)
    } else {
        (
            None,
            Some(checkpoint.load_parameter_schema(client, model.schema())?),
        )
    };
    let load_time = load_start.elapsed();
    let residency_pool = if let Some(offloaded) = &offloaded {
        let (_, device) = cuda_offload_memories(client)?;
        Some(
            WeightResidencyPool::new(device, offloaded.payload_bytes())
                .with_allocator_headroom(options.offload_headroom_bytes),
        )
    } else {
        None
    };
    let prefetch_start = Instant::now();
    let prefetched = match (&offloaded, &residency_pool) {
        (Some(offloaded), Some(pool)) => Some(pool.prefetch(offloaded)?),
        (None, None) => None,
        _ => return Err("invalid transformer residency pool state".into()),
    };
    let prefetch_time = prefetch_start.elapsed();
    let compile_start = Instant::now();
    let mut compiler =
        compiler(client, options.cache_dir, "qwen-image-2.1-dit-v1")?.with_cudnn_attention(true);
    if !options.device_loop {
        compiler = compiler.with_compute_dtype(rxla_core::DType::BF16);
    }
    let compiled = model.compile(&mut compiler)?;
    let compile_time = compile_start.elapsed();
    let prefetch_wait_start = Instant::now();
    if let Some(prefetched) = &prefetched {
        prefetched.block_until_ready()?;
    }
    let prefetch_wait_time = prefetch_wait_start.elapsed();
    let (offloaded_runner, device_runner) = match (prefetched, resident) {
        (Some(prefetched), None) => (Some(prefetched.bind(&compiled)?), None),
        (None, Some(resident)) => (None, Some(compiled.bind_parameters(resident.bindings())?)),
        _ => return Err("invalid transformer residency state".into()),
    };
    let context = client.buffer(&prompt.shape, &prompt.values)?;
    let mut random = ChaCha8Rng::seed_from_u64(options.seed);
    let initial = (0..image_tokens * 64)
        .map(|_| StandardNormal.sample(&mut random))
        .collect::<Vec<f32>>();
    let mut sample = client.buffer(&[1, image_tokens, 64], &initial)?;
    let execute_start = Instant::now();
    if options.device_loop {
        sample = match (&offloaded_runner, &device_runner) {
            (Some(runner), None) => runner.run((&sample, &context))?,
            (None, Some(runner)) => runner.run((&sample, &context))?,
            _ => return Err("invalid transformer runner state".into()),
        };
    } else {
        for index in 0..schedule.len() {
            let step = schedule.step(index)?;
            let timestep = client.buffer(&[1], &[step.timestep])?;
            let step_size = client.buffer(&[1], &[step.step_size])?;
            sample = match (&offloaded_runner, &device_runner) {
                (Some(runner), None) => runner.run((&sample, &timestep, &step_size, &context))?,
                (None, Some(runner)) => runner.run((&sample, &timestep, &step_size, &context))?,
                _ => return Err("invalid transformer runner state".into()),
            };
        }
    }
    let execute_time = execute_start.elapsed();
    eprintln!(
        "denoise mode={} residency={} trace={trace_time:.3?} weights={load_time:.3?} compile={compile_time:.3?} prefetch-submit={prefetch_time:.3?} prefetch-tail={prefetch_wait_time:.3?} execute={execute_time:.3?} steps={}",
        if options.device_loop {
            "device-loop"
        } else {
            "host-enqueue"
        },
        if options.offload_transformer {
            "pinned-host"
        } else {
            "device"
        },
        options.steps,
    );
    Ok(sample.to_vec::<f32>()?)
}

fn decode(
    client: &Client,
    checkpoint_path: &PathBuf,
    latent: &[f32],
    latent_height: i64,
    latent_width: i64,
    cache_dir: Option<&PathBuf>,
) -> Result<Vec<f32>> {
    let image_tokens = latent_height * latent_width;
    let model = Model::new(|cx: Cx, latent| {
        qwen_image_2_1_vae_decoder(cx, &latent, latent_height, latent_width)
    })
    .inputs(ModelInput::new([1, image_tokens, 64]))
    .trace()?;
    let mut checkpoint = SafeTensors::open(checkpoint_path)?;
    let parameters = checkpoint.load_parameter_schema(client, model.schema())?;
    let compiled = model.compile(&mut compiler(client, cache_dir, "qwen-image-2.1-vae-v1")?)?;
    let runner = compiled.bind_parameters(parameters.bindings())?;
    let latent = client.buffer(&[1, image_tokens, 64], latent)?;
    let image: Buffer = runner.run(&latent)?;
    Ok(image.to_vec::<f32>()?)
}

fn tile_starts(length: usize, tile: usize, stride: usize) -> Vec<usize> {
    if length <= tile {
        return vec![0];
    }
    let mut starts = (0..=length - tile).step_by(stride).collect::<Vec<_>>();
    let final_start = length - tile;
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    starts
}

fn axis_weight(position: usize, index: usize, starts: &[usize], tile: usize) -> f32 {
    let mut weight = 1.0_f32;
    if index > 0 {
        let overlap = starts[index - 1] + tile - starts[index];
        if position < overlap {
            weight = weight.min((position + 1) as f32 / (overlap + 1) as f32);
        }
    }
    if index + 1 < starts.len() {
        let overlap = starts[index] + tile - starts[index + 1];
        if position >= tile - overlap {
            weight = weight.min((tile - position) as f32 / (overlap + 1) as f32);
        }
    }
    weight
}

#[allow(clippy::too_many_arguments)]
fn decode_tiled(
    client: &Client,
    checkpoint_path: &PathBuf,
    latent: &[f32],
    latent_height: usize,
    latent_width: usize,
    tile_height: usize,
    tile_width: usize,
    stride_height: usize,
    stride_width: usize,
    cache_dir: Option<&PathBuf>,
) -> Result<Vec<f32>> {
    const CHANNELS: usize = 64;
    const SCALE: usize = 16;
    const OUTPUT_CHANNELS: usize = 4;
    let tile_height_i64 = i64::try_from(tile_height)?;
    let tile_width_i64 = i64::try_from(tile_width)?;
    let tile_tokens_i64 = i64::try_from(tile_height * tile_width)?;
    let model = Model::new(|cx: Cx, latent| {
        qwen_image_2_1_vae_decoder_raw(cx, &latent, tile_height_i64, tile_width_i64)
    })
    .inputs(ModelInput::new([1, tile_tokens_i64, 64]))
    .trace()?;
    let mut checkpoint = SafeTensors::open(checkpoint_path)?;
    let parameters = checkpoint.load_parameter_schema(client, model.schema())?;
    let compiled = model.compile(&mut compiler(
        client,
        cache_dir,
        "qwen-image-2.1-vae-tiled-v1",
    )?)?;
    let runner = compiled.bind_parameters(parameters.bindings())?;
    let y_starts = tile_starts(latent_height, tile_height, stride_height);
    let x_starts = tile_starts(latent_width, tile_width, stride_width);
    let output_height = latent_height * SCALE;
    let output_width = latent_width * SCALE;
    let mut accumulated = vec![0.0_f32; output_height * output_width * OUTPUT_CHANNELS];
    let mut weights = vec![0.0_f32; output_height * output_width];
    let output_tile_height = tile_height * SCALE;
    let output_tile_width = tile_width * SCALE;
    let output_y_starts = y_starts
        .iter()
        .map(|start| start * SCALE)
        .collect::<Vec<_>>();
    let output_x_starts = x_starts
        .iter()
        .map(|start| start * SCALE)
        .collect::<Vec<_>>();
    for (tile_y, &start_y) in y_starts.iter().enumerate() {
        for (tile_x, &start_x) in x_starts.iter().enumerate() {
            let mut tile = vec![0.0_f32; tile_height * tile_width * CHANNELS];
            for y in 0..tile_height {
                let source = ((start_y + y) * latent_width + start_x) * CHANNELS;
                let target = y * tile_width * CHANNELS;
                tile[target..target + tile_width * CHANNELS]
                    .copy_from_slice(&latent[source..source + tile_width * CHANNELS]);
            }
            let tile = client.buffer(&[1, i64::try_from(tile_height * tile_width)?, 64], &tile)?;
            let decoded: Buffer = runner.run(&tile)?;
            let decoded = decoded.to_vec::<f32>()?;
            for y in 0..output_tile_height {
                let wy = axis_weight(y, tile_y, &output_y_starts, output_tile_height);
                let output_y = start_y * SCALE + y;
                for x in 0..output_tile_width {
                    let wx = axis_weight(x, tile_x, &output_x_starts, output_tile_width);
                    let weight = wy * wx;
                    let output_x = start_x * SCALE + x;
                    let output_pixel = output_y * output_width + output_x;
                    let tile_pixel = y * output_tile_width + x;
                    weights[output_pixel] += weight;
                    for channel in 0..OUTPUT_CHANNELS {
                        accumulated[output_pixel * OUTPUT_CHANNELS + channel] +=
                            decoded[tile_pixel * OUTPUT_CHANNELS + channel] * weight;
                    }
                }
            }
            eprintln!(
                "vae_tile={}/{} origin={}x{}",
                tile_y * x_starts.len() + tile_x + 1,
                y_starts.len() * x_starts.len(),
                start_x * SCALE,
                start_y * SCALE,
            );
        }
    }
    for (pixel, &weight) in accumulated
        .as_chunks_mut::<OUTPUT_CHANNELS>()
        .0
        .iter_mut()
        .zip(&weights)
    {
        if weight == 0.0 {
            return Err("VAE tiling left an uncovered output pixel".into());
        }
        for channel in pixel {
            *channel /= weight;
        }
    }
    Ok(accumulated)
}

fn run(args: Args) -> Result<()> {
    if args.width == 0
        || args.height == 0
        || !args.width.is_multiple_of(16)
        || !args.height.is_multiple_of(16)
    {
        return Err("width and height must be positive multiples of 16".into());
    }
    if args.vae_tile == 0
        || !args.vae_tile.is_multiple_of(16)
        || args.vae_overlap >= args.vae_tile
        || !args.vae_overlap.is_multiple_of(16)
    {
        return Err("VAE tile and overlap must be multiples of 16, with tile > overlap".into());
    }
    if args
        .memory_fraction
        .is_some_and(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
    {
        return Err("memory fraction must be finite and between zero and one".into());
    }
    let mut client_options = ClientOptions::new().set("preallocate", args.preallocate);
    if let Some(memory_fraction) = args.memory_fraction {
        client_options = client_options.set("memory_fraction", memory_fraction);
    }
    let client = unsafe { Client::load_with_options(&args.plugin, &client_options) }?;
    let latent_height = i64::from(args.height / 16);
    let latent_width = i64::from(args.width / 16);
    let mut latent = denoise(
        &client,
        &DenoiseOptions {
            checkpoint_path: &args.transformer,
            conditioning_path: &args.conditioning,
            latent_height,
            latent_width,
            steps: args.steps,
            seed: args.seed,
            device_loop: args.device_loop,
            cache_dir: args.cache_dir.as_ref(),
            offload_transformer: args.offload_transformer,
            offload_headroom_bytes: args
                .offload_headroom_mib
                .checked_mul(1024 * 1024)
                .ok_or("offload headroom exceeds usize")?,
        },
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
    let values = if args.width <= args.vae_tile && args.height <= args.vae_tile {
        decode(
            &client,
            &args.vae,
            &latent,
            latent_height,
            latent_width,
            args.cache_dir.as_ref(),
        )?
    } else {
        let tile_height = usize::try_from((args.vae_tile / 16).min(args.height / 16))?;
        let tile_width = usize::try_from((args.vae_tile / 16).min(args.width / 16))?;
        let stride = usize::try_from((args.vae_tile - args.vae_overlap) / 16)?;
        decode_tiled(
            &client,
            &args.vae,
            &latent,
            usize::try_from(latent_height)?,
            usize::try_from(latent_width)?,
            tile_height,
            tile_width,
            stride.min(tile_height),
            stride.min(tile_width),
            args.cache_dir.as_ref(),
        )?
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feathered_tiles_cover_irregular_final_stride_with_unit_weight() {
        let starts = tile_starts(128, 64, 48);
        assert_eq!(starts, [0, 48, 64]);
        let mut weights = [0.0_f32; 128];
        for (index, &start) in starts.iter().enumerate() {
            for position in 0..64 {
                weights[start + position] += axis_weight(position, index, &starts, 64);
            }
        }
        for weight in weights {
            assert!((weight - 1.0).abs() < 1e-6, "partition weight {weight}");
        }
    }
}
