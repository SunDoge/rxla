use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::flux2_autoencoder_decoder;
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Validate and execute the official FLUX.2 VAE decoder")]
struct Args {
    /// Official vae/diffusion_pytorch_model.safetensors checkpoint.
    weights: PathBuf,
    /// Trusted PJRT plugin. Omit for header/schema validation only.
    #[arg(long)]
    plugin: Option<PathBuf>,
    #[arg(long, default_value_t = 2)]
    height: i64,
    #[arg(long, default_value_t = 2)]
    width: i64,
}

fn run(args: Args) -> Result<()> {
    if args.height <= 0 || args.width <= 0 {
        return Err("latent grid dimensions must be positive".into());
    }
    let tokens = args.height * args.width;
    let model =
        Model::new(|cx: Cx, input| flux2_autoencoder_decoder(cx, &input, args.height, args.width))
            .inputs(ModelInput::new([1, tokens, 128]))
            .trace()?;
    let mut checkpoint = SafeTensors::open(args.weights)?;
    checkpoint.validate_parameter_schema(model.schema())?;
    println!(
        "schema_ok parameters={} output_shape={:?}",
        model.schema().parameters().len(),
        model.outputs()[0].shape(),
    );
    let Some(plugin) = args.plugin else {
        return Ok(());
    };
    let client = unsafe { Client::load(plugin) }?;
    let weights = checkpoint.load_parameter_schema(&client, model.schema())?;
    let compile_started = std::time::Instant::now();
    let compiled = model.compile(&mut Compiler::new(client.clone(), CacheLimits::default()))?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;
    let runner = compiled.bind_parameters(weights.bindings())?;
    let input = client.buffer(&[1, tokens, 128], &vec![0.0_f32; (tokens * 128) as usize])?;
    let started = std::time::Instant::now();
    let output: Buffer = runner.run(&input)?;
    let values = output.to_vec::<f32>()?;
    println!(
        "backend={} compile_ms={compile_ms:.3} elapsed_ms={:.3} checksum={:.9}",
        client.info()?.platform,
        started.elapsed().as_secs_f64() * 1e3,
        values.iter().map(|&value| f64::from(value)).sum::<f64>(),
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
