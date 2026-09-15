//! Real quantized Qwen3.5-0.8B text-prefill benchmark.
use clap::Parser;
use rxla_core::{CacheLimits, Client, Compiler, DType};
use rxla_models::{Qwen3_5Config, qwen3_5};
use rxla_nn::{Cx, Model};
use rxla_safetensors::SafeTensors;
use std::{io::Write, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Benchmark quantized Qwen3.5-0.8B text prefill")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// SafeTensors produced by benchmarks/quantize_qwen3_5_w8.py.
    weights: PathBuf,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(i64).range(1..=4096))]
    sequence: i64,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 10, value_parser = parse_positive_usize)]
    iterations: usize,
    /// Optional little-endian F32 logits for oracle comparison.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("invalid iteration count: {error}"))?;
    (value > 0)
        .then_some(value)
        .ok_or_else(|| "iteration count must be positive".into())
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

fn run(args: Args) -> Result<()> {
    let client = unsafe { Client::load(&args.plugin) }?;
    let info = client.info()?;
    let config = Qwen3_5Config::qwen3_5_0_8b_w8();
    let model = Model::new(|cx: &mut Cx| {
        let ids = cx.input_dtype(&[1, args.sequence], DType::I32)?;
        qwen3_5(cx, &ids, &config)
    });
    let (schema, applied) = model.trace()?;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    checkpoint.require_metadata(&[("format", "rxla-qwen3.5-w8-v1"), ("group_size", "128")])?;
    let load_start = Instant::now();
    let weights = checkpoint.load_parameter_schema(&client, &schema)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compile_start = Instant::now();
    let executable = applied.compile(&mut compiler)?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1e3;
    let token_values = (0..args.sequence)
        .map(|index| 1000 + index as i32)
        .collect::<Vec<_>>();
    let tokens = client.buffer(&[1, args.sequence], &token_values)?;
    let arguments = applied.bind(&[&tokens], weights.bindings())?;
    for _ in 0..args.warmup {
        executable.execute(arguments.as_slice())?;
    }
    let mut samples = Vec::with_capacity(args.iterations);
    let mut output = Vec::new();
    for _ in 0..args.iterations {
        let start = Instant::now();
        output = executable.execute(arguments.as_slice())?;
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let logits = output[0].to_vec::<f32>()?;
    let checksum = logits.iter().map(|&value| f64::from(value)).sum::<f64>();
    if let Some(path) = args.output {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for value in &logits {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }
    println!(
        "backend={} sequence={} parameters={} compile_ms={compile_ms:.3} load_ms={load_ms:.3} warmup={} iterations={} mean_ms={mean:.3} p50_ms={:.3} p95_ms={:.3} tokens_per_second={:.3} checksum={checksum:.9}",
        info.platform,
        args.sequence,
        schema.parameters().len(),
        args.warmup,
        args.iterations,
        percentile(&samples, 0.50),
        percentile(&samples, 0.95),
        args.sequence as f64 / (mean / 1e3),
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
