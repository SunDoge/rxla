//! Benchmark one real conditional Stable Diffusion UNet evaluation.
use clap::Parser;
use rxla_core::{CacheLimits, Client, Compiler};
use rxla_models::{UnetConfig, unet};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::{io::Write, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Benchmark one real conditional Stable Diffusion UNet evaluation")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Diffusers UNet safetensors checkpoint.
    weights: PathBuf,
    #[arg(default_value_t = 64, value_parser = parse_spatial)]
    spatial: i64,
    #[arg(default_value_t = 2)]
    warmup: usize,
    #[arg(default_value_t = 10, value_parser = parse_positive_usize)]
    iterations: usize,
    output: Option<PathBuf>,
}

fn parse_spatial(value: &str) -> std::result::Result<i64, String> {
    let value = value
        .parse::<i64>()
        .map_err(|error| format!("invalid spatial size: {error}"))?;
    if value <= 0 || value % 2 != 0 {
        return Err("spatial size must be positive and even".into());
    }
    Ok(value)
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("invalid iteration count: {error}"))?;
    if value == 0 {
        return Err("iteration count must be positive".into());
    }
    Ok(value)
}

fn timestep_embedding(timestep: f32, width: usize) -> Vec<f32> {
    let half = width / 2;
    let frequencies =
        (0..half).map(|index| (-10_000_f32.ln() * index as f32 / half as f32).exp() * timestep);
    let angles = frequencies.collect::<Vec<_>>();
    angles
        .iter()
        .map(|value| value.cos())
        .chain(angles.iter().map(|value| value.sin()))
        .collect()
}

fn values(count: i64, modulus: i64, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index % modulus) as f32 - (modulus / 2) as f32) / scale)
        .collect()
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

fn run(args: Args) -> Result<()> {
    let client = unsafe { Client::load(&args.plugin) }?;
    let info = client.info()?;
    let model = Model::new(|cx: &mut Cx, sample, timestep, context| {
        unet(cx, &sample, &timestep, &context, &UnetConfig::tiny())
    })
    .inputs((
        ModelInput::new([1, args.spatial, args.spatial, 4]),
        ModelInput::new([1, 32]),
        ModelInput::new([1, 77, 32]),
    ));
    let (schema, applied) = model.trace()?;

    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compile_start = Instant::now();
    let executable = applied.compile(&mut compiler)?;
    let compile_time = compile_start.elapsed();
    let load_start = Instant::now();
    let weights = checkpoint.load_parameter_schema(&client, &schema)?;
    let load_time = load_start.elapsed();

    let sample_values = values(args.spatial * args.spatial * 4, 251, 64.);
    let context_values = values(77 * 32, 241, 128.);
    let timestep_values = timestep_embedding(501., 32);
    let sample = client.buffer(&[1, args.spatial, args.spatial, 4], &sample_values)?;
    let timestep = client.buffer(&[1, 32], &timestep_values)?;
    let context = client.buffer(&[1, 77, 32], &context_values)?;
    let arguments = applied.bind((&sample, &timestep, &context), weights.bindings())?;
    for _ in 0..args.warmup {
        executable.execute(arguments.as_slice())?;
    }

    let mut samples = Vec::with_capacity(args.iterations);
    let mut result = Vec::new();
    for _ in 0..args.iterations {
        let start = Instant::now();
        result = executable.execute(arguments.as_slice())?;
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let download_start = Instant::now();
    let output = result[0].to_vec::<f32>()?;
    let download_time = download_start.elapsed();
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let checksum = output.iter().map(|&value| f64::from(value)).sum::<f64>();
    if let Some(path) = args.output {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for value in &output {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }
    println!(
        "backend={} device={} spatial={} parameters={} compile_ms={:.3} load_ms={:.3} warmup={} iterations={} mean_ms={mean:.3} p50_ms={:.3} p95_ms={:.3} output_download_ms={:.3} checksum={checksum:.9}",
        info.platform,
        info.addressable_devices
            .iter()
            .find(|device| device.selected)
            .map_or("unknown", |device| device.kind.as_str()),
        args.spatial,
        schema.parameters().len(),
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
