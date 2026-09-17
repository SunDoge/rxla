//! Benchmark the native SAM 2.1 Hiera-small image trunk on PJRT.

use clap::{Parser, ValueEnum};
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::sam2::{HieraConfig, hiera};
use rxla_nn::{Cx, Model, ModelInput, path};
use rxla_safetensors::SafeTensors;
use std::{path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Benchmark the native SAM 2.1 Hiera-small image trunk")]
struct Args {
    /// Trusted CUDA PJRT plugin shared library.
    plugin: PathBuf,
    /// Converted official SAM 2.1 safetensors checkpoint.
    weights: PathBuf,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    /// Internal CUDA compute precision; inputs and outputs remain F32.
    #[arg(long, value_enum, default_value_t = ComputePrecision::F16)]
    precision: ComputePrecision,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComputePrecision {
    F32,
    Bf16,
    F16,
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

fn run(args: Args) -> Result<()> {
    if args.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let client = unsafe { Client::load(&args.plugin) }?;
    let info = client.info()?;
    let parameter_dtype = match args.precision {
        ComputePrecision::F32 => rxla_core::DType::F32,
        ComputePrecision::Bf16 => rxla_core::DType::BF16,
        ComputePrecision::F16 => rxla_core::DType::F16,
    };
    let definition = Model::new(move |cx: Cx, image: rxla_core::Tensor| {
        hiera(
            path!(cx / "image_encoder" / "trunk")?,
            &image,
            &HieraConfig::sam2_1_small().parameter_dtype(parameter_dtype),
        )
    })
    .inputs(ModelInput::new([1, 1024, 1024, 3]))
    .trace()?;
    let compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut compiler = match args.precision {
        ComputePrecision::F32 => compiler,
        ComputePrecision::Bf16 => compiler.with_compute_dtype(rxla_core::DType::BF16),
        ComputePrecision::F16 => compiler.with_compute_dtype(rxla_core::DType::F16),
    };
    let compile_started = Instant::now();
    let compiled = definition.compile(&mut compiler)?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;

    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let load_started = Instant::now();
    let weights = checkpoint.load_parameter_schema(&client, definition.schema())?;
    let runner = compiled.bind_parameters(weights.bindings())?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let input = client.buffer(&[1, 1024, 1024, 3], &vec![0.0_f32; 1024 * 1024 * 3])?;

    for _ in 0..args.warmup {
        let _: [Buffer; 4] = runner.run(&input)?;
    }
    let mut samples = Vec::with_capacity(args.iterations);
    let mut outputs = None;
    for _ in 0..args.iterations {
        let started = Instant::now();
        outputs = Some(runner.run::<_, [Buffer; 4]>(&input)?);
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let checksum = outputs
        .expect("positive iteration count checked")
        .into_iter()
        .map(|output| {
            output
                .to_vec::<f32>()
                .map(|values| values.into_iter().map(f64::from).sum::<f64>())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<f64>();
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!(
        "backend={} device={} precision={:?} model=sam2.1-hiera-small batch=1 image=1024 compile_ms={compile_ms:.3} load_ms={load_ms:.3} warmup={} iterations={} mean_ms={mean:.3} p50_ms={:.3} p95_ms={:.3} checksum={checksum:.9}",
        info.platform,
        info.addressable_devices
            .iter()
            .find(|device| device.selected)
            .map_or("unknown", |device| device.kind.as_str()),
        args.precision,
        args.warmup,
        args.iterations,
        percentile(&samples, 0.50),
        percentile(&samples, 0.95),
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
