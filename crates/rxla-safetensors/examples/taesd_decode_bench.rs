//! Benchmark the real Stable Diffusion TAESD decoder on a PJRT backend.
use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::taesd_decoder;
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::{io::Write, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Benchmark a TAESD decoder on a PJRT backend")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Diffusers-compatible TAESD safetensors checkpoint.
    weights: PathBuf,
    #[arg(default_value_t = 64, value_parser = parse_positive_i64)]
    latent: i64,
    #[arg(default_value_t = 2)]
    warmup: usize,
    #[arg(default_value_t = 10, value_parser = parse_positive_usize)]
    iterations: usize,
    /// Optional raw F32 output path.
    output: Option<PathBuf>,
}

fn parse_positive_i64(value: &str) -> std::result::Result<i64, String> {
    match value.parse() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err("must be a positive integer".into()),
    }
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    match value.parse() {
        Ok(value) if value > 0 => Ok(value),
        _ => Err("must be a positive integer".into()),
    }
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * fraction).round() as usize;
    samples[index]
}

fn run(args: Args) -> Result<()> {
    let client = unsafe { Client::load(&args.plugin) }?;
    let info = client.info()?;
    let decoder = Model::new(|cx: Cx, latent| taesd_decoder(cx, &latent))
        .inputs(ModelInput::new([1, args.latent, args.latent, 4]));
    let decoder = decoder.trace()?;
    let schema = decoder.schema();

    let checkpoint_open = Instant::now();
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let checkpoint_open = checkpoint_open.elapsed();

    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compile_start = Instant::now();
    let compiled = decoder.compile(&mut compiler)?;
    let compile_time = compile_start.elapsed();

    let load_start = Instant::now();
    let weights = checkpoint.load_parameter_schema(&client, &schema)?;
    let bindings = weights.bindings();
    let runner = compiled.bind_parameters(bindings)?;
    let load_time = load_start.elapsed();

    let values = (0..args.latent * args.latent * 4)
        .map(|index| ((index % 257) as f32 - 128.) / 64.)
        .collect::<Vec<_>>();
    let latent = client.buffer(&[1, args.latent, args.latent, 4], &values)?;
    for _ in 0..args.warmup {
        let _: Buffer = runner.run(&latent)?;
    }

    let mut samples = Vec::with_capacity(args.iterations);
    let mut result = None;
    for _ in 0..args.iterations {
        let start = Instant::now();
        result = Some(runner.run::<_, Buffer>(&latent)?);
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let download_start = Instant::now();
    let final_image = result
        .expect("at least one measured iteration")
        .to_vec::<f32>()?;
    let download_time = download_start.elapsed();
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let checksum = final_image
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>();
    let squared = final_image
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();

    if let Some(path) = args.output {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for value in &final_image {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }
    println!(
        "backend={} device={} latent={} image={} checkpoint_open_ms={:.3} compile_ms={:.3} load_ms={:.3} warmup={} iterations={} mean_ms={mean:.3} p50_ms={:.3} p95_ms={:.3} output_download_ms={:.3} checksum={checksum:.9} squared_sum={squared:.9}",
        info.platform,
        info.addressable_devices
            .iter()
            .find(|device| device.selected)
            .map_or("unknown", |device| device.kind.as_str()),
        args.latent,
        args.latent * 8,
        checkpoint_open.as_secs_f64() * 1e3,
        compile_time.as_secs_f64() * 1e3,
        load_time.as_secs_f64() * 1e3,
        args.warmup,
        args.iterations,
        percentile(&samples, 0.5),
        percentile(&samples, 0.95),
        download_time.as_secs_f64() * 1e3,
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
