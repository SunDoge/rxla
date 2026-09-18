use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler, DType};
use rxla_models::flux2::{Flux2Config, flux2};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Validate and benchmark the official FLUX.2 Klein 4B transformer")]
struct Args {
    /// Official flux-2-klein-4b.safetensors checkpoint.
    weights: PathBuf,
    /// Trusted PJRT plugin. Omit to perform header/schema validation only.
    #[arg(long)]
    plugin: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    image_tokens: i64,
    #[arg(long, default_value_t = 1)]
    context_tokens: i64,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
}

fn run(args: Args) -> Result<()> {
    if args.image_tokens <= 0 || args.context_tokens <= 0 {
        return Err("token counts must be positive".into());
    }
    let config = Flux2Config::klein_4b();
    let model = Model::new(|cx: Cx, image, image_ids, timestep, context, context_ids| {
        flux2(
            cx,
            &image,
            &image_ids,
            &timestep,
            &context,
            &context_ids,
            &config,
        )
    })
    .inputs((
        ModelInput::new([1, args.image_tokens, 128]),
        ModelInput::new([1, args.image_tokens, 4]),
        ModelInput::new([1]),
        ModelInput::new([1, args.context_tokens, 7680]),
        ModelInput::new([1, args.context_tokens, 4]),
    ))
    .trace()?;
    let mut checkpoint = SafeTensors::open(args.weights)?;
    checkpoint.validate_parameter_schema(model.schema())?;
    println!(
        "schema_ok parameters={} elements={}",
        model.schema().parameters().len(),
        model
            .schema()
            .parameters()
            .iter()
            .map(|parameter| parameter.shape().iter().product::<i64>())
            .sum::<i64>()
    );
    let Some(plugin) = args.plugin else {
        return Ok(());
    };
    let client = unsafe { Client::load(plugin) }?;
    if args.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let load_started = std::time::Instant::now();
    let weights = checkpoint.load_parameter_schema(&client, model.schema())?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let load_stats = checkpoint.stats();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default())
        .with_compute_dtype(DType::BF16)
        .with_cudnn_attention(true);
    let compile_started = std::time::Instant::now();
    let compiled = model.compile(&mut compiler)?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;
    let runner = compiled.bind_parameters(weights.bindings())?;
    let image = client.buffer(
        &[1, args.image_tokens, 128],
        &vec![0.0_f32; (args.image_tokens * 128) as usize],
    )?;
    let image_ids = client.buffer(
        &[1, args.image_tokens, 4],
        &vec![0.0_f32; (args.image_tokens * 4) as usize],
    )?;
    let timestep = client.buffer(&[1], &[1.0_f32])?;
    let context = client.buffer(
        &[1, args.context_tokens, 7680],
        &vec![0.0_f32; (args.context_tokens * 7680) as usize],
    )?;
    let context_ids = client.buffer(
        &[1, args.context_tokens, 4],
        &vec![0.0_f32; (args.context_tokens * 4) as usize],
    )?;
    for _ in 0..args.warmup {
        let output: Buffer = runner.run((&image, &image_ids, &timestep, &context, &context_ids))?;
        output.to_vec::<f32>()?;
    }
    let mut durations = Vec::with_capacity(args.iterations);
    let mut values = Vec::new();
    for _ in 0..args.iterations {
        let started = std::time::Instant::now();
        let output: Buffer = runner.run((&image, &image_ids, &timestep, &context, &context_ids))?;
        values = output.to_vec::<f32>()?;
        durations.push(started.elapsed().as_secs_f64() * 1e3);
    }
    durations.sort_by(f64::total_cmp);
    let mean_ms = durations.iter().sum::<f64>() / durations.len() as f64;
    println!(
        "backend={} image_tokens={} context_tokens={} load_ms={load_ms:.3} read_ms={:.3} decode_ms={:.3} upload_ms={:.3} compile_ms={compile_ms:.3} warmup={} iterations={} mean_ms={mean_ms:.3} p50_ms={:.3} checksum={:.9}",
        client.info()?.platform,
        args.image_tokens,
        args.context_tokens,
        load_stats.read_time.as_secs_f64() * 1e3,
        load_stats.decode_time.as_secs_f64() * 1e3,
        load_stats.upload_time.as_secs_f64() * 1e3,
        args.warmup,
        args.iterations,
        durations[(durations.len() - 1) / 2],
        values.iter().map(|&value| f64::from(value)).sum::<f64>(),
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}
