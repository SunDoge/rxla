use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::qwen_image_2_1::qwen_image_2_1_vae_decoder;
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Compare the RXLA Qwen-Image-2.1 VAE with Diffusers")]
struct Args {
    plugin: PathBuf,
    weights: PathBuf,
    oracle: PathBuf,
}

fn run(args: Args) -> Result<()> {
    let client = unsafe { Client::load(args.plugin) }?;
    let model = Model::new(|cx: Cx, latent| qwen_image_2_1_vae_decoder(cx, &latent, 1, 1))
        .inputs(ModelInput::new([1, 1, 64]))
        .trace()?;
    let mut weights = SafeTensors::open(args.weights)?;
    let parameters = weights.load_parameter_schema(&client, model.schema())?;
    let compiled = model.compile(&mut Compiler::new(client.clone(), CacheLimits::default()))?;
    let runner = compiled.bind_parameters(parameters.bindings())?;
    let mut oracle = SafeTensors::open(args.oracle)?;
    let latent = oracle.read_f32("latent")?;
    let latent = client.buffer(&latent.shape, &latent.values)?;
    let output: Buffer = runner.run(&latent)?;
    let actual = output.to_vec::<f32>()?;
    let expected = oracle.read_f32("output")?.values;
    let max_absolute = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let mean_absolute = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| f64::from((actual - expected).abs()))
        .sum::<f64>()
        / actual.len() as f64;
    println!(
        "backend={} max_absolute={max_absolute:.9} mean_absolute={mean_absolute:.9} checksum={:.9}",
        client.info()?.platform,
        actual.iter().map(|&value| f64::from(value)).sum::<f64>(),
    );
    if max_absolute > 2e-4 {
        return Err(format!(
            "Qwen-Image-2.1 VAE numerical mismatch: max absolute error {max_absolute}"
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
