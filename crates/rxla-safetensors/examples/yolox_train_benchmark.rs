//! Benchmark a repeatable YOLOX device training step against the Torch oracle.

use clap::{Parser, ValueEnum};
use rxla_core::{Buffer, CacheLimits, Client, Compiler, DType, Precision, Tensor};
use rxla_models::{
    YoloConfig, YoloLossConfig, YoloSimOtaConfig, YoloTargets, yolo, yolox_loss_with_l1_weight,
    yolox_simota,
};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use rxla_train::{SgdOptions, apply_model_sgd_with_learning_rate};
use std::{path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const CLASSES: usize = 80;
const MAX_TARGETS: usize = 120;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComputeDtype {
    F32,
    Bf16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ConvolutionPrecision {
    Default,
    Highest,
}

#[derive(Parser)]
#[command(about = "Benchmark a synthetic, resident YOLOX training step")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Checkpoint produced by benchmarks/convert_yolox.py.
    weights: PathBuf,
    #[arg(long, default_value_t = 416)]
    size: i64,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    #[arg(long, default_value_t = 20)]
    iterations: usize,
    #[arg(long, default_value_t = 12)]
    targets_per_image: usize,
    #[arg(long, default_value_t = 0.00015625)]
    learning_rate: f32,
    #[arg(long, value_enum, default_value_t = ComputeDtype::F32)]
    compute_dtype: ComputeDtype,
    #[arg(long, value_enum, default_value_t = ConvolutionPrecision::Default)]
    convolution_precision: ConvolutionPrecision,
    #[arg(long)]
    l1: bool,
    /// Bracket timed iterations with the CUDA profiler API.
    #[arg(long)]
    cuda_profile: bool,
    #[arg(long, default_value = "libcudart.so.13")]
    cuda_runtime: PathBuf,
}

struct CudaProfiler {
    _library: libloading::Library,
    stop: unsafe extern "C" fn() -> i32,
}

impl CudaProfiler {
    fn start(path: &std::path::Path) -> Result<Self> {
        // SAFETY: the library remains loaded while the copied CUDA Runtime
        // function pointers are used, and both have a stable C ABI.
        let library = unsafe { libloading::Library::new(path) }?;
        let start =
            unsafe { *library.get::<unsafe extern "C" fn() -> i32>(b"cudaProfilerStart\0")? };
        let stop = unsafe { *library.get::<unsafe extern "C" fn() -> i32>(b"cudaProfilerStop\0")? };
        let status = unsafe { start() };
        if status != 0 {
            return Err(format!("cudaProfilerStart failed with status {status}").into());
        }
        Ok(Self {
            _library: library,
            stop,
        })
    }
}

impl Drop for CudaProfiler {
    fn drop(&mut self) {
        // SAFETY: `_library` keeps the symbol alive through this call.
        let _ = unsafe { (self.stop)() };
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size <= 0
        || args.size % 32 != 0
        || args.batch_size == 0
        || args.iterations == 0
        || !(1..=MAX_TARGETS).contains(&args.targets_per_image)
        || !args.learning_rate.is_finite()
        || args.learning_rate < 0.0
    {
        return Err("invalid benchmark arguments".into());
    }
    let batch = i64::try_from(args.batch_size)?;
    let compute_dtype = match args.compute_dtype {
        ComputeDtype::F32 => DType::F32,
        ComputeDtype::Bf16 => DType::BF16,
    };
    let convolution_precision = match args.convolution_precision {
        ConvolutionPrecision::Default => Precision::Default,
        ConvolutionPrecision::Highest => Precision::Highest,
    };
    let config = YoloConfig::yolox_tiny(CLASSES as i64)
        .compute_dtype(compute_dtype)
        .convolution_precision(convolution_precision);
    let definition = Model::new(
        move |cx: Cx,
              image: Tensor,
              target_boxes: Tensor,
              target_classes: Tensor,
              target_valid: Tensor,
              learning_rate: Tensor,
              l1_weight: Tensor| {
            let heads = yolo(cx, &image.cast(DType::F32)?, &config)?;
            let assignments = yolox_simota(
                &heads,
                &YoloTargets {
                    boxes: target_boxes,
                    classes: target_classes,
                    valid: target_valid,
                },
                YoloSimOtaConfig::default(),
            )?;
            let loss = yolox_loss_with_l1_weight(
                &heads,
                &assignments,
                YoloLossConfig::default(),
                &l1_weight,
            )?;
            Ok::<_, rxla_models::Error>((loss.total, learning_rate))
        },
    )
    .inputs((
        ModelInput::new([batch, args.size, args.size, 3]).with_dtype(DType::U8),
        ModelInput::new([batch, MAX_TARGETS as i64, 4]),
        ModelInput::new([batch, MAX_TARGETS as i64, CLASSES as i64]),
        ModelInput::new([batch, MAX_TARGETS as i64, 1]),
        ModelInput::new([]),
        ModelInput::new([]),
    ))
    .training();
    let (trainable, mut model) = definition.trace_resident_all()?;
    let loss = model.outputs()[0].clone();
    let learning_rate = model.outputs()[1].clone();
    apply_model_sgd_with_learning_rate(
        &mut model,
        &trainable,
        &loss,
        &learning_rate,
        |parameter| {
            let decay =
                parameter.path().ends_with(".weight") && !parameter.path().contains(".bn.weight");
            SgdOptions::default()
                .momentum(0.9)
                .nesterov(true)
                .weight_decay(if decay { 5e-4 } else { 0.0 })
        },
    )?;

    let client = unsafe { Client::load(&args.plugin) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compile_started = Instant::now();
    let program = model.compile_stateful_tensors(&mut compiler, &[loss])?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    checkpoint.require_metadata(&[("format", "rxla-yolox-v1")])?;
    let weights = checkpoint.load_model_schema(&client, model.schema())?;
    let mut session = weights.initialize_raw(model.session(&program))?.build()?;
    let inputs = synthetic_inputs(&client, &args)?;

    let mut loss = f32::NAN;
    for _ in 0..args.warmup {
        loss = run_step(&mut session, &inputs)?;
    }
    let profiler = args
        .cuda_profile
        .then(|| CudaProfiler::start(&args.cuda_runtime))
        .transpose()?;
    let mut samples = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let started = Instant::now();
        loss = run_step(&mut session, &inputs)?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    drop(profiler);
    let total_ms = samples.iter().sum::<f64>();
    println!(
        "rxla precision={} convolution_precision={} size={} batch={} iterations={} targets_per_image={} l1={} loss={loss:.6}",
        match args.compute_dtype {
            ComputeDtype::F32 => "f32",
            ComputeDtype::Bf16 => "bf16",
        },
        match args.convolution_precision {
            ConvolutionPrecision::Default => "default",
            ConvolutionPrecision::Highest => "highest",
        },
        args.size,
        args.batch_size,
        args.iterations,
        args.targets_per_image,
        args.l1,
    );
    println!(
        "mean_ms={:.3} p50_ms={:.3} p95_ms={:.3} images_per_second={:.1} compile_ms={compile_ms:.1}",
        total_ms / samples.len() as f64,
        percentile(&samples, 0.5),
        percentile(&samples, 0.95),
        args.batch_size as f64 * args.iterations as f64 / (total_ms / 1e3),
    );
    Ok(())
}

struct Inputs {
    image: Buffer,
    boxes: Buffer,
    classes: Buffer,
    valid: Buffer,
    learning_rate: Buffer,
    l1_weight: Buffer,
}

fn synthetic_inputs(client: &Client, args: &Args) -> Result<Inputs> {
    let batch = i64::try_from(args.batch_size)?;
    let elements = args.batch_size * usize::try_from(args.size * args.size * 3)?;
    let image = (0..elements)
        .map(|index| ((index * 17 + 13) % 256) as u8)
        .collect::<Vec<_>>();
    let mut boxes = vec![0.0_f32; args.batch_size * MAX_TARGETS * 4];
    let mut classes = vec![0.0_f32; args.batch_size * MAX_TARGETS * CLASSES];
    let mut valid = vec![0.0_f32; args.batch_size * MAX_TARGETS];
    let scale = args.size as f32 / 64.0;
    for batch_index in 0..args.batch_size {
        for target in 0..args.targets_per_image {
            let slot = batch_index * MAX_TARGETS + target;
            boxes[slot * 4..slot * 4 + 4].copy_from_slice(&[
                (8 + target * 11 % 48) as f32 * scale,
                (8 + target * 17 % 48) as f32 * scale,
                (6 + target % 13) as f32 * scale,
                (7 + target * 3 % 12) as f32 * scale,
            ]);
            classes[slot * CLASSES + target % CLASSES] = 1.0;
            valid[slot] = 1.0;
        }
    }
    Ok(Inputs {
        image: client.buffer(&[batch, args.size, args.size, 3], &image)?,
        boxes: client.buffer(&[batch, MAX_TARGETS as i64, 4], &boxes)?,
        classes: client.buffer(&[batch, MAX_TARGETS as i64, CLASSES as i64], &classes)?,
        valid: client.buffer(&[batch, MAX_TARGETS as i64, 1], &valid)?,
        learning_rate: client.buffer(&[], &[args.learning_rate])?,
        l1_weight: client.buffer(&[], &[f32::from(args.l1)])?,
    })
}

fn run_step(session: &mut rxla_core::Session, inputs: &Inputs) -> Result<f32> {
    let outputs = session.run(&[
        &inputs.image,
        &inputs.boxes,
        &inputs.classes,
        &inputs.valid,
        &inputs.learning_rate,
        &inputs.l1_weight,
    ])?;
    Ok(outputs[0].to_vec::<f32>()?[0])
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}
