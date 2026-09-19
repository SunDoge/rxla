//! Train YOLOX on real COCO samples with device SimOTA and resident momentum SGD.

use clap::{Parser, Subcommand, ValueEnum};
use fast_image_resize::{
    FilterType as FastFilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image as FastImage, ImageRef},
};
use fory::{ForyRow, from_row};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
#[cfg(feature = "disk-cache")]
use rxla_core::DiskCache;
use rxla_core::{
    CacheLimits, Client, ClientOptions, Compiler, DType, Precision, StateProgram, Tensor,
};
use rxla_log::{RunLog, proto as log_proto};
use rxla_models::{
    YoloConfig, YoloLossConfig, YoloSimOtaConfig, YoloTargets, yolo, yolox_loss_with_l1_weight,
    yolox_simota,
};
use rxla_nn::{AppliedModel, Cx, Model, ModelInput};
use rxla_safetensors::{SafeTensors, save_buffers_new, save_model_new, save_training_state_new};
use rxla_tfrecord::Reader as TfrecordReader;
use rxla_train::{
    BoundedPipeline, DataRng, DetectionTarget, ModelEma, SampleRng, SgdOptions, ShufflePool,
    YoloLrSchedule, apply_model_ema, apply_model_ema_when,
    apply_model_sgd_accumulating_with_learning_rate, apply_model_sgd_with_learning_rate,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MAX_TARGETS: usize = 120;
const BASIC_LEARNING_RATE_PER_IMAGE: f32 = 0.01 / 64.0;

fn default_learning_rate(batch_size: usize) -> f32 {
    BASIC_LEARNING_RATE_PER_IMAGE * batch_size as f32
}

fn default_augmentation_workers() -> usize {
    // A batch already fans out over Rayon's global pool. One ordered producer
    // keeps that pool busy and avoids multiplying in-flight image buffers.
    1
}

fn default_prefetch() -> usize {
    default_augmentation_workers() * 8
}

fn official_epoch_schedule(
    samples: usize,
    effective_batch: usize,
    epochs: usize,
) -> Option<(usize, usize, usize, usize)> {
    if samples == 0 || effective_batch == 0 || epochs == 0 {
        return None;
    }
    let steps_per_epoch = samples.div_ceil(effective_batch);
    Some((
        steps_per_epoch.checked_mul(epochs)?,
        steps_per_epoch.checked_mul(5)?,
        steps_per_epoch.checked_mul(16)?,
        steps_per_epoch.checked_mul(15)?,
    ))
}

#[derive(Parser)]
#[command(about = "Run consecutive real COCO YOLOX training steps")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Optional initial checkpoint produced by benchmarks/convert_yolox.py.
    /// Omit it (and --resume) to initialize the model from its declared policies.
    weights: Option<PathBuf>,
    #[command(subcommand)]
    source: DataSource,
    /// Total optimizer steps in the global training schedule.
    #[arg(long, default_value_t = 1)]
    steps: usize,
    /// Derive total, warmup, no-augmentation, and minimum-rate steps from the
    /// source sample count and effective batch using the official epoch recipe.
    #[arg(long, conflicts_with = "steps")]
    epochs: Option<usize>,
    /// Limit updates in this invocation while retaining the global --steps schedule.
    #[arg(long)]
    run_steps: Option<usize>,
    #[arg(long, default_value_t = 416)]
    size: i64,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Microbatches averaged before each optimizer and EMA update.
    #[arg(long, default_value_t = 1)]
    gradient_accumulation: usize,
    /// Per-image padded target capacity; official YOLOX uses 120.
    #[arg(long, default_value_t = MAX_TARGETS)]
    target_capacity: usize,
    /// Whether to use YOLOX-tiny's random 320..640 input size every 10 steps.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    random_resize: bool,
    #[arg(long, default_value_t = 10)]
    random_size_min: u32,
    #[arg(long, default_value_t = 20)]
    random_size_max: u32,
    /// Number of prepared host batches retained ahead of the device.
    #[arg(long, default_value_t = default_prefetch())]
    prefetch: usize,
    /// Concurrent ordered host batches submitted to the augmentation pool.
    #[arg(long, default_value_t = default_augmentation_workers())]
    augmentation_workers: usize,
    /// Number of sequential samples retained for deterministic pool shuffle.
    #[arg(long, default_value_t = 1024)]
    shuffle_buffer: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 1.0)]
    mosaic_probability: f32,
    /// Official YOLOX-tiny disables MixUp; set to 1 for the base recipe.
    #[arg(long, default_value_t = 0.0)]
    mixup_probability: f32,
    /// Total optimizer learning rate; defaults to YOLOX's 0.01/64 per image
    /// multiplied by this process's batch size.
    #[arg(long)]
    learning_rate: Option<f32>,
    /// Internal CUDA arithmetic; model inputs, parameters, state, and outputs remain F32.
    #[arg(long, value_enum, default_value_t = ComputeDtype::F32)]
    compute_dtype: ComputeDtype,
    /// StableHLO convolution precision; default permits accelerator-native TF32.
    #[arg(long, value_enum, default_value_t = ConvolutionPrecision::Default)]
    convolution_precision: ConvolutionPrecision,
    /// Disable XLA CUDA command buffers and launch kernels directly.
    #[arg(long)]
    direct_launches: bool,
    /// Trusted directory for backend-native executable caching.
    #[arg(long)]
    compile_cache: Option<PathBuf>,
    /// PJRT allocator reservation fraction; omit to use the plugin default.
    #[arg(long)]
    memory_fraction: Option<f32>,
    /// Whether PJRT preallocates its allocator reservation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    preallocate: bool,
    /// Disable the official YOLOX parameter EMA, primarily for profiling.
    #[arg(long)]
    disable_model_ema: bool,
    /// Override warmup duration (default: 5/300 of all steps).
    #[arg(long)]
    warmup_steps: Option<usize>,
    /// Override the Mosaic-off/L1 tail (step mode defaults to 15/300).
    #[arg(long)]
    no_augmentation_steps: Option<usize>,
    /// Override the minimum-learning-rate tail independently of no-augmentation.
    #[arg(long)]
    minimum_rate_steps: Option<usize>,
    /// Write inference-ready trained parameters and BatchNorm state here.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Resume parameters, BatchNorm, SGD momentum, and EMA from a training snapshot.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Write a resumable snapshot including optimizer and EMA state here.
    #[arg(long)]
    training_output: Option<PathBuf>,
    /// Atomically refresh --training-output every N completed optimizer steps.
    #[arg(long, requires = "training_output")]
    checkpoint_every: Option<usize>,
    /// Write structured Protobuf events, indexes, and attachments to a new MCAP file.
    #[arg(long)]
    event_log: Option<PathBuf>,
    /// Stable run identity recorded in the event log; generated when omitted.
    #[arg(long, requires = "event_log")]
    run_id: Option<String>,
    /// Export every compiled multi-scale training variant as StableHLO into a new directory.
    #[arg(long)]
    stablehlo_dir: Option<PathBuf>,
    /// Synchronize, validate, and print metrics every N completed steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Bracket steady training with the CUDA profiler API (for nsys capture-range).
    #[arg(long)]
    cuda_profile: bool,
    /// CUDA runtime library used by --cuda-profile.
    #[arg(long, default_value = "libcudart.so.13")]
    cuda_runtime: PathBuf,
}

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

#[derive(Clone, Debug, Subcommand)]
enum DataSource {
    /// Read individual JPEG files and COCO instances JSON.
    Coco {
        annotations: PathBuf,
        images: PathBuf,
        /// COCO image ID at which to start iterating.
        image_id: u64,
    },
    /// Stream TFRecord-framed Apache Fory Standard Rows.
    Tfrecord { input: PathBuf },
}

#[derive(Deserialize)]
struct Coco {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct CocoAnnotation {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
    area: f32,
    #[serde(default)]
    iscrowd: u8,
}

#[derive(Deserialize)]
struct CocoCategory {
    id: u64,
}

#[derive(Deserialize)]
struct ForyManifest {
    codec: String,
    container: String,
    #[serde(default)]
    annotation_policy: String,
    schema_sha256: String,
    samples: usize,
}

#[derive(ForyRow)]
struct DetectionRecord {
    // PyFory Standard Row canonicalizes dataclass fields lexically.
    boxes: Vec<f32>,
    height: i32,
    image: Vec<u8>,
    labels: Vec<i32>,
    sample_id: String,
    width: i32,
}

struct DetectionSample {
    id: String,
    encoded_image: Vec<u8>,
    dimensions: Option<[u32; 2]>,
    boxes: Vec<[f32; 4]>,
    classes: Vec<usize>,
}

struct PreparedBatch {
    sample_ids: Vec<String>,
    pixels: Vec<u8>,
    targets: Vec<DetectionTarget>,
    size: u32,
    load_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TargetPaddingStats {
    maximum_per_image: usize,
    dropped: usize,
}

struct PaddedTargets {
    boxes: Vec<f32>,
    classes: Vec<i32>,
    valid: Vec<f32>,
    stats: TargetPaddingStats,
}

struct TrainingVariant {
    model: AppliedModel,
    program: StateProgram,
    ema: Option<ModelEma>,
}

#[derive(Clone, Copy)]
struct TrainingVariantConfig {
    batch: i64,
    augmentation_size: i64,
    size: i64,
    target_capacity: i64,
    model: YoloConfig,
    use_ema: bool,
    gradient_accumulation: usize,
    export_stablehlo: bool,
}

struct BuiltTrainingVariant {
    variant: TrainingVariant,
    stablehlo: Option<Vec<u8>>,
}

struct CudaProfiler {
    _library: libloading::Library,
    stop: unsafe extern "C" fn() -> i32,
}

impl CudaProfiler {
    fn start(path: &std::path::Path) -> Result<Self> {
        // SAFETY: the library is retained for the lifetime of the copied
        // function pointers and both symbols have CUDA Runtime's stable C ABI.
        let library = unsafe { libloading::Library::new(path) }?;
        let start =
            unsafe { *library.get::<unsafe extern "C" fn() -> i32>(b"cudaProfilerStart\0")? };
        let stop = unsafe { *library.get::<unsafe extern "C" fn() -> i32>(b"cudaProfilerStop\0")? };
        // SAFETY: no arguments or borrowed state cross the CUDA C ABI.
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
        // SAFETY: the retained library keeps this CUDA C symbol valid.
        let _ = unsafe { (self.stop)() };
    }
}

struct AugmentationGroup {
    samples: Vec<DetectionSample>,
    mixup: Option<DetectionSample>,
    rng: SampleRng,
}

struct AugmentationBatch {
    groups: Vec<AugmentationGroup>,
    size: u32,
}

struct SourceSamples(TrainingSource);

struct PlannedBatches<I> {
    samples: I,
    remaining: usize,
    step: usize,
    mosaic_steps: usize,
    mosaic_probability: f32,
    mixup_probability: f32,
    batch_size: usize,
    sizes: Vec<u32>,
    rng: DataRng,
}

enum TrainingSource {
    Coco {
        document: Coco,
        images: PathBuf,
        position: usize,
        rollover_position: usize,
        annotations: HashMap<u64, Vec<usize>>,
        categories: BTreeMap<u64, usize>,
    },
    Tfrecord {
        reader: TfrecordReader<std::io::BufReader<std::fs::File>>,
        input: PathBuf,
        categories: BTreeMap<u64, usize>,
        samples: usize,
    },
}

fn training_sizes(
    base: u32,
    steps: usize,
    seed: u64,
    random_resize: bool,
    minimum: u32,
    maximum: u32,
) -> Vec<u32> {
    let rng = DataRng::new(seed);
    let choices = u64::from(maximum - minimum + 1);
    let mut current = base;
    (0..steps)
        .map(|step| {
            if random_resize && step != 0 && step % 10 == 0 {
                let block = (step / 10) as u64;
                let multiple = minimum + (rng.sample(0, block).word(91, 0) % choices) as u32;
                current = multiple * 32;
            }
            current
        })
        .collect()
}

fn build_training_variant(
    compiler: &mut Compiler,
    config: TrainingVariantConfig,
) -> Result<BuiltTrainingVariant> {
    let TrainingVariantConfig {
        batch,
        augmentation_size,
        size,
        target_capacity,
        model: model_config,
        use_ema,
        gradient_accumulation,
        export_stablehlo,
    } = config;
    let definition = Model::new(
        move |cx: Cx,
              image: Tensor,
              target_boxes: Tensor,
              target_classes: Tensor,
              target_valid: Tensor,
              learning_rate: Tensor,
              l1_weight: Tensor| {
            let mut image = image.cast(DType::F32)?;
            let mut target_boxes = target_boxes;
            if size != augmentation_size {
                image = image.resize_bilinear2d([size, size])?;
                target_boxes = target_boxes.mul_scalar(size as f32 / augmentation_size as f32)?;
            }
            let outputs = yolo(cx, &image, &model_config)?;
            let targets = YoloTargets {
                boxes: target_boxes,
                class_indices: target_classes,
                valid: target_valid,
            };
            let assignments = yolox_simota(&outputs, &targets, YoloSimOtaConfig::default())?;
            let foreground = assignments.objectness.sum([0, 1, 2], false)?;
            let loss = yolox_loss_with_l1_weight(
                &outputs,
                &assignments,
                YoloLossConfig::default(),
                &l1_weight,
            )?;
            Ok::<_, rxla_models::Error>((
                loss.total,
                loss.boxes,
                loss.objectness,
                loss.classes,
                loss.l1,
                foreground,
                learning_rate,
            ))
        },
    )
    .inputs((
        ModelInput::new([batch, augmentation_size, augmentation_size, 3]).with_dtype(DType::U8),
        ModelInput::new([batch, target_capacity, 4]),
        ModelInput::new([batch, target_capacity, 1]).with_dtype(DType::I32),
        ModelInput::new([batch, target_capacity, 1]),
        ModelInput::new([]),
        ModelInput::new([]),
    ))
    .training();
    let (trainable, mut model) = definition.trace_resident_all()?;
    let metrics = Tensor::stack(&model.outputs()[..6], 0)?;
    let loss = model.outputs()[0].clone();
    let learning_rate = model.outputs()[6].clone();
    let options = |parameter: &rxla_nn::ParameterSpec| {
        let decay =
            parameter.path().ends_with(".weight") && !parameter.path().contains(".bn.weight");
        SgdOptions::default()
            .momentum(0.9)
            .nesterov(true)
            .weight_decay(if decay { 5e-4 } else { 0.0 })
    };
    let accumulation = if gradient_accumulation == 1 {
        apply_model_sgd_with_learning_rate(&mut model, &trainable, &loss, &learning_rate, options)?;
        None
    } else {
        Some(apply_model_sgd_accumulating_with_learning_rate(
            &mut model,
            &trainable,
            &loss,
            &learning_rate,
            gradient_accumulation,
            options,
        )?)
    };
    if let Some(accumulation) = &accumulation {
        model.gate_model_state_updates_where(accumulation.applied(), |state| {
            state.path().ends_with(".running_mean") || state.path().ends_with(".running_variance")
        })?;
    }
    let ema = if use_ema {
        Some(if let Some(accumulation) = &accumulation {
            apply_model_ema_when(&mut model, &trainable, 0.9998, accumulation.applied())?
        } else {
            apply_model_ema(&mut model, &trainable, 0.9998)?
        })
    } else {
        None
    };
    let prepared = model.prepare_stateful_tensors(&[metrics])?;
    let stablehlo = export_stablehlo.then(|| prepared.lowered_program().code().to_vec());
    let program = prepared.compile(compiler)?;
    Ok(BuiltTrainingVariant {
        variant: TrainingVariant {
            model,
            program,
            ema,
        },
        stablehlo,
    })
}

fn training_metadata(
    args: &Args,
    total_steps: usize,
    base_learning_rate: f32,
    warmup_steps: usize,
    no_augmentation_steps: usize,
    minimum_rate_steps: usize,
    completed_steps: usize,
) -> HashMap<String, String> {
    [
        ("format", "rxla-yolox-training-v3".to_owned()),
        ("completed_steps", completed_steps.to_string()),
        ("total_steps", total_steps.to_string()),
        ("batch_size", args.batch_size.to_string()),
        (
            "gradient_accumulation",
            args.gradient_accumulation.to_string(),
        ),
        ("target_capacity", args.target_capacity.to_string()),
        ("base_learning_rate", base_learning_rate.to_string()),
        ("warmup_steps", warmup_steps.to_string()),
        ("no_augmentation_steps", no_augmentation_steps.to_string()),
        ("minimum_rate_steps", minimum_rate_steps.to_string()),
        ("size", args.size.to_string()),
        ("random_resize", args.random_resize.to_string()),
        ("random_size_min", args.random_size_min.to_string()),
        ("random_size_max", args.random_size_max.to_string()),
        ("seed", args.seed.to_string()),
        ("shuffle_buffer", args.shuffle_buffer.to_string()),
        ("mosaic_probability", args.mosaic_probability.to_string()),
        ("mixup_probability", args.mixup_probability.to_string()),
        ("compute_dtype", format!("{:?}", args.compute_dtype)),
        (
            "convolution_precision",
            format!("{:?}", args.convolution_precision),
        ),
        ("model_ema", (!args.disable_model_ema).to_string()),
        ("source", format!("{:?}", args.source)),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}

fn periodic_checkpoint_path(path: &Path, completed_steps: usize) -> PathBuf {
    let mut file_name = path.file_stem().unwrap_or_default().to_os_string();
    file_name.push(format!(".step-{completed_steps}"));
    if let Some(extension) = path.extension() {
        file_name.push(".");
        file_name.push(extension);
    }
    path.with_file_name(file_name)
}

fn unix_time_ns() -> Result<u64> {
    Ok(u64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
    )?)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut output = File::options().write(true).create_new(true).open(path)?;
    output.write_all(bytes)?;
    output.flush()?;
    Ok(())
}

fn run_started_event(
    run_id: &str,
    args: &Args,
    total_steps: usize,
    resumed_at: usize,
    effective_batch_size: usize,
) -> log_proto::RunStarted {
    let metadata = [
        log_proto::Metric::integer("total_steps", total_steps as i64),
        log_proto::Metric::integer("resumed_at", resumed_at as i64),
        log_proto::Metric::integer("batch_size", args.batch_size as i64),
        log_proto::Metric::integer("gradient_accumulation", args.gradient_accumulation as i64),
        log_proto::Metric::integer("effective_batch_size", effective_batch_size as i64),
        log_proto::Metric::integer("seed", args.seed as i64),
        log_proto::Metric::text("compute_dtype", format!("{:?}", args.compute_dtype)),
        log_proto::Metric::text(
            "convolution_precision",
            format!("{:?}", args.convolution_precision),
        ),
        log_proto::Metric::text("source", format!("{:?}", args.source)),
    ]
    .into_iter()
    .collect();
    log_proto::RunStarted {
        run_id: run_id.to_owned(),
        model: "yolox-tiny".to_owned(),
        producer: concat!("rxla/", env!("CARGO_PKG_VERSION")).to_owned(),
        metadata,
    }
}

fn resume_step(
    args: &Args,
    total_steps: usize,
    base_learning_rate: f32,
    warmup_steps: usize,
    no_augmentation_steps: usize,
    minimum_rate_steps: usize,
) -> Result<usize> {
    let Some(path) = &args.resume else {
        return Ok(0);
    };
    let checkpoint = SafeTensors::open(path)?;
    let metadata = checkpoint
        .metadata()
        .ok_or("training snapshot has no metadata")?;
    let completed = metadata
        .get("completed_steps")
        .ok_or("training snapshot has no completed_steps")?
        .parse::<usize>()?;
    if completed > total_steps {
        return Err(format!(
            "training snapshot step {completed} exceeds requested total {}",
            total_steps
        )
        .into());
    }
    let mut expected = training_metadata(
        args,
        total_steps,
        base_learning_rate,
        warmup_steps,
        no_augmentation_steps,
        minimum_rate_steps,
        completed,
    );
    expected.remove("completed_steps");
    for (key, value) in expected {
        if metadata.get(&key) != Some(&value) {
            return Err(
                format!("training snapshot metadata {key:?} does not match this run").into(),
            );
        }
    }
    Ok(completed)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size <= 0
        || args.size % 32 != 0
        || args
            .learning_rate
            .is_some_and(|learning_rate| !learning_rate.is_finite() || learning_rate < 0.0)
        || args.steps == 0
        || args
            .memory_fraction
            .is_some_and(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
        || args.epochs == Some(0)
        || args.run_steps == Some(0)
        || args.checkpoint_every == Some(0)
        || args.log_every == 0
        || args.batch_size == 0
        || args.gradient_accumulation == 0
        || !(1..=MAX_TARGETS).contains(&args.target_capacity)
        || args.prefetch == 0
        || args.augmentation_workers == 0
        || args.shuffle_buffer == 0
        || args.random_size_min == 0
        || args.random_size_min > args.random_size_max
        || args.random_size_max > u32::MAX / 32
        || !(0.0..=1.0).contains(&args.mosaic_probability)
        || !(0.0..=1.0).contains(&args.mixup_probability)
        || args.output.is_some() && args.output == args.training_output
    {
        return Err(
            "invalid training configuration: size must be a positive multiple of 32; rates/probabilities and counts must be in range; output paths must differ"
                .into(),
        );
    }
    let effective_batch_size = args
        .batch_size
        .checked_mul(args.gradient_accumulation)
        .ok_or("effective batch size overflow")?;
    let source = TrainingSource::open(args.source.clone())?;
    let source_sample_count = source.sample_count();
    if source_sample_count == 0 {
        return Err("training source is empty".into());
    }
    let steps_per_epoch = source_sample_count.div_ceil(effective_batch_size);
    let epoch_schedule = args
        .epochs
        .map(|epochs| {
            official_epoch_schedule(source_sample_count, effective_batch_size, epochs)
                .ok_or("invalid or overflowing epoch-derived training schedule")
        })
        .transpose()?;
    let total_steps = epoch_schedule.map_or(args.steps, |schedule| schedule.0);
    let optimizer_step_sizes = training_sizes(
        u32::try_from(args.size)?,
        total_steps,
        args.seed,
        args.random_resize,
        args.random_size_min,
        args.random_size_max,
    );
    let total_micro_steps = total_steps
        .checked_mul(args.gradient_accumulation)
        .ok_or("training microstep count overflow")?;
    let step_sizes = optimizer_step_sizes
        .into_iter()
        .flat_map(|size| std::iter::repeat_n(size, args.gradient_accumulation))
        .collect::<Vec<_>>();
    let base_learning_rate = args
        .learning_rate
        .unwrap_or_else(|| default_learning_rate(effective_batch_size));
    let warmup_steps = args
        .warmup_steps
        .unwrap_or_else(|| epoch_schedule.map_or(total_steps / 60, |schedule| schedule.1));
    let no_augmentation_steps = args
        .no_augmentation_steps
        .unwrap_or_else(|| epoch_schedule.map_or(total_steps / 20, |schedule| schedule.2));
    let minimum_rate_steps = args
        .minimum_rate_steps
        .unwrap_or_else(|| epoch_schedule.map_or(total_steps / 20, |schedule| schedule.3));
    if warmup_steps + minimum_rate_steps > total_steps || no_augmentation_steps > total_steps {
        return Err("warmup, no-augmentation, or minimum-rate steps exceed total steps".into());
    }
    let resumed_at = resume_step(
        &args,
        total_steps,
        base_learning_rate,
        warmup_steps,
        no_augmentation_steps,
        minimum_rate_steps,
    )?;
    let event_log_started_ns = unix_time_ns()?;
    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| format!("yolox-{event_log_started_ns}"));
    let mut event_log = args.event_log.as_ref().map(RunLog::create).transpose()?;
    if let Some(log) = &mut event_log {
        log.write(
            &run_started_event(
                &run_id,
                &args,
                total_steps,
                resumed_at,
                effective_batch_size,
            ),
            event_log_started_ns,
        )?;
        log.flush()?;
    }
    let samples = ShufflePool::new(SourceSamples(source), args.shuffle_buffer, args.seed)?;
    let mut batches = PlannedBatches {
        samples,
        remaining: total_micro_steps,
        step: 0,
        mosaic_steps: total_steps
            .saturating_sub(no_augmentation_steps)
            .saturating_mul(args.gradient_accumulation),
        mosaic_probability: args.mosaic_probability,
        mixup_probability: args.mixup_probability,
        batch_size: args.batch_size,
        sizes: step_sizes.clone(),
        rng: DataRng::new(args.seed),
    };
    let replay_started = Instant::now();
    let replay_micro_steps = resumed_at
        .checked_mul(args.gradient_accumulation)
        .ok_or("resume microstep count overflow")?;
    for step in 0..replay_micro_steps {
        batches
            .next()
            .ok_or_else(|| format!("training data ended while replaying step {step}"))??;
    }
    if let Some(run_steps) = args.run_steps {
        let run_micro_steps = run_steps
            .checked_mul(args.gradient_accumulation)
            .ok_or("run microstep count overflow")?;
        batches.remaining = batches.remaining.min(run_micro_steps);
    }
    let expected_micro_steps = replay_micro_steps + batches.remaining;
    let expected_completed = expected_micro_steps / args.gradient_accumulation;
    let replay_ms = replay_started.elapsed().as_secs_f64() * 1e3;
    // Host input preparation starts before XLA compilation so initial pool fill
    // and the first augmented batches are normally hidden by compilation.
    let augmentation_size = u32::try_from(args.size)?;
    let pipeline = BoundedPipeline::from_iter(args.prefetch, batches)?.map_parallel_ordered(
        args.prefetch,
        args.augmentation_workers,
        move |batch| prepare_batch(batch.groups, augmentation_size, batch.size),
    )?;
    let batch = i64::try_from(args.batch_size)?;
    let mut client_options = ClientOptions::new().set("preallocate", args.preallocate);
    if let Some(memory_fraction) = args.memory_fraction {
        client_options = client_options.set("memory_fraction", memory_fraction);
    }
    let client = unsafe { Client::load_with_options(&args.plugin, &client_options) }?;
    let compiler = Compiler::new(client.clone(), CacheLimits::default());
    #[cfg(feature = "disk-cache")]
    let compiler = if let Some(directory) = &args.compile_cache {
        // SAFETY: native cache entries are executable code. The CLI makes the
        // trust boundary explicit and never enables a directory implicitly.
        let cache = unsafe {
            DiskCache::new_for_client(directory, "rxla-yolox-train-v1", &client, 512 * 1024 * 1024)?
        };
        compiler.with_disk_cache(cache)
    } else {
        compiler
    };
    #[cfg(not(feature = "disk-cache"))]
    let compiler = {
        if args.compile_cache.is_some() {
            return Err("--compile-cache requires the disk-cache cargo feature".into());
        }
        compiler
    };
    let mut compiler = if args.direct_launches {
        compiler.without_cuda_command_buffers()
    } else {
        compiler
    };
    let compute_dtype = match args.compute_dtype {
        ComputeDtype::F32 => DType::F32,
        ComputeDtype::Bf16 => DType::BF16,
    };
    let convolution_precision = match args.convolution_precision {
        ConvolutionPrecision::Default => Precision::Default,
        ConvolutionPrecision::Highest => Precision::Highest,
    };
    let training_config = YoloConfig::yolox_tiny(80)
        .compute_dtype(compute_dtype)
        .convolution_precision(convolution_precision);
    let train_compile_started = Instant::now();
    let mut variants = BTreeMap::new();
    let active_sizes = if replay_micro_steps < expected_micro_steps {
        &step_sizes[replay_micro_steps..expected_micro_steps]
    } else {
        &step_sizes[total_micro_steps - 1..]
    };
    for size in active_sizes.iter().copied().collect::<BTreeSet<_>>() {
        let built = build_training_variant(
            &mut compiler,
            TrainingVariantConfig {
                batch,
                augmentation_size: args.size,
                size: i64::from(size),
                target_capacity: i64::try_from(args.target_capacity)?,
                model: training_config,
                use_ema: !args.disable_model_ema,
                gradient_accumulation: args.gradient_accumulation,
                export_stablehlo: args.stablehlo_dir.is_some(),
            },
        )?;
        if let (Some(directory), Some(stablehlo)) = (&args.stablehlo_dir, built.stablehlo) {
            fs::create_dir_all(directory)?;
            write_new(&directory.join(format!("train-{size}.mlir")), &stablehlo)?;
        }
        variants.insert(size, built.variant);
    }
    if let Some(directory) = &args.stablehlo_dir {
        let manifest = variants
            .keys()
            .map(|size| format!("{size}\ttrain-{size}.mlir\n"))
            .collect::<String>();
        write_new(&directory.join("variants.tsv"), manifest.as_bytes())?;
    }
    let train_compile_ms = train_compile_started.elapsed().as_secs_f64() * 1e3;
    let mut current_size = step_sizes[replay_micro_steps.min(total_micro_steps - 1)];
    let initial = variants
        .get(&current_size)
        .expect("the first scheduled size was compiled");
    let initial_checkpoint = args.resume.as_ref().or(args.weights.as_ref());
    let template = if let Some(initial_checkpoint) = initial_checkpoint {
        let mut checkpoint = SafeTensors::open(initial_checkpoint)?;
        let weights = checkpoint.load_model_schema(&client, initial.model.schema())?;
        weights
            .initialize_raw(initial.model.session(&initial.program))?
            .initializer_seed(args.seed)
            .build()?
    } else {
        initial
            .model
            .session(&initial.program)
            .initialize_parameters(args.seed)?
            .build()?
    };
    let mut training_session = if let Some(path) = &args.resume {
        let mut checkpoint = SafeTensors::open(path)?;
        checkpoint
            .load_training_state(&client, &initial.model, &template)?
            .initialize_raw(initial.model.session(&initial.program))?
            .build()?
    } else {
        template
    };
    let schedule = YoloLrSchedule::new(base_learning_rate, total_steps)
        .warmup_steps(warmup_steps)
        .minimum_rate_steps(minimum_rate_steps);
    let l1_weights = [
        client.buffer(&[], &[0.0_f32])?,
        client.buffer(&[], &[1.0_f32])?,
    ];
    let mut completed = resumed_at;
    let mut completed_micro_steps = replay_micro_steps;
    let mut maximum_targets_per_image = 0;
    let mut dropped_targets = 0;
    let mut warned_about_dropped_targets = false;
    let mut total_wait_ms = 0.0;
    let mut total_load_ms = 0.0;
    let mut total_staging_ms = 0.0;
    let mut total_image_upload_ms = 0.0;
    let mut total_train_ms = 0.0;
    let run_started = Instant::now();
    let mut profiler = None;
    loop {
        let wait_started = Instant::now();
        let Some(prepared) = pipeline.recv() else {
            break;
        };
        let wait_ms = wait_started.elapsed().as_secs_f64() * 1e3;
        let PreparedBatch {
            sample_ids,
            pixels,
            targets,
            size,
            load_ms,
        } = prepared?;
        total_wait_ms += wait_ms;
        total_load_ms += load_ms;
        if size != current_size {
            let snapshot = variants
                .get(&current_size)
                .expect("current training size was compiled")
                .model
                .take_session(training_session)?;
            let target = variants
                .get(&size)
                .expect("scheduled training size was compiled");
            training_session = snapshot
                .restore_model(target.model.session(&target.program))?
                .build()?;
            current_size = size;
        }
        let staging_started = Instant::now();
        let input = client.buffer(&[batch, args.size, args.size, 3], &pixels)?;
        let image_upload_ms = staging_started.elapsed().as_secs_f64() * 1e3;
        // Official YOLOX updates the optimizer's LR after `optimizer.step()`.
        // Therefore the first step uses the optimizer's initial warmup rate,
        // and a step with `completed` preceding updates uses schedule position
        // `completed` (the rate installed after the previous update).
        let step_learning_rate = schedule.at_update(completed)?;
        let learning_rate = client.buffer(&[], &[step_learning_rate])?;
        let use_l1 = no_augmentation_steps != 0
            && completed >= total_steps.saturating_sub(no_augmentation_steps);
        let l1_weight = &l1_weights[usize::from(use_l1)];
        let padded = padded_targets(args.batch_size, args.target_capacity, 80, &targets)?;
        let target_stats = padded.stats;
        maximum_targets_per_image = maximum_targets_per_image.max(target_stats.maximum_per_image);
        dropped_targets += target_stats.dropped;
        if target_stats.dropped != 0 && !warned_about_dropped_targets {
            eprintln!(
                "warning: --target-capacity={} dropped {} targets at step {}; observed {} targets in one image",
                args.target_capacity,
                target_stats.dropped,
                completed + 1,
                target_stats.maximum_per_image,
            );
            warned_about_dropped_targets = true;
        }
        let capacity = i64::try_from(args.target_capacity)?;
        let target_boxes = client.buffer(&[batch, capacity, 4], &padded.boxes)?;
        let target_classes = client.buffer(&[batch, capacity, 1], &padded.classes)?;
        let target_valid = client.buffer(&[batch, capacity, 1], &padded.valid)?;
        let staging_ms = staging_started.elapsed().as_secs_f64() * 1e3;
        total_staging_ms += staging_ms;
        total_image_upload_ms += image_upload_ms;
        let train_started = Instant::now();
        let outputs = training_session.run(&[
            &input,
            &target_boxes,
            &target_classes,
            &target_valid,
            &learning_rate,
            l1_weight,
        ])?;
        completed_micro_steps += 1;
        let optimizer_step = completed_micro_steps % args.gradient_accumulation == 0;
        if optimizer_step {
            completed += 1;
        }
        let observe = optimizer_step
            && (completed == resumed_at + 1
                || completed % args.log_every == 0
                || completed == expected_completed);
        let losses = if observe {
            let losses = outputs[0].to_vec::<f32>()?;
            if losses.len() != 6 || losses.iter().any(|value| !value.is_finite()) {
                return Err("training produced invalid losses".into());
            }
            Some(losses)
        } else {
            None
        };
        let train_ms = train_started.elapsed().as_secs_f64() * 1e3;
        total_train_ms += train_ms;
        if optimizer_step && completed == resumed_at + 1 && args.cuda_profile {
            profiler = Some(CudaProfiler::start(&args.cuda_runtime)?);
        }
        if let Some(losses) = losses {
            let foreground = losses[5] as usize;
            println!(
                "step={completed} size={size} image_id={} targets={} max_targets={} dropped_targets={} foreground={} total={:.6} box={:.6} objectness={:.6} classes={:.6} l1={:.6} lr={step_learning_rate:.8} wait_ms={wait_ms:.1} load_ms={load_ms:.1} staging_ms={staging_ms:.1} image_upload_ms={image_upload_ms:.1} train_ms={train_ms:.1}",
                sample_ids.join(","),
                targets.len(),
                target_stats.maximum_per_image,
                target_stats.dropped,
                foreground,
                losses[0],
                losses[1],
                losses[2],
                losses[3],
                losses[4],
            );
            if let Some(log) = &mut event_log {
                let mut event = log_proto::TrainStep::new(completed as u64, f64::from(losses[0]))
                    .learning_rate(f64::from(step_learning_rate));
                event.epoch = u64::try_from(completed / steps_per_epoch)?;
                event.samples_seen = u64::try_from(
                    completed
                        .checked_mul(effective_batch_size)
                        .ok_or("sample count overflow")?,
                )?;
                event.elapsed_ns = u64::try_from(run_started.elapsed().as_nanos())?;
                event.metrics = vec![
                    log_proto::Metric::scalar("box_loss", f64::from(losses[1])),
                    log_proto::Metric::scalar("objectness_loss", f64::from(losses[2])),
                    log_proto::Metric::scalar("class_loss", f64::from(losses[3])),
                    log_proto::Metric::scalar("l1_loss", f64::from(losses[4])),
                    log_proto::Metric::integer("foreground", foreground as i64),
                    log_proto::Metric::integer("targets", targets.len() as i64),
                    log_proto::Metric::integer("input_size", i64::from(size)),
                    log_proto::Metric::scalar("wait_ms", wait_ms),
                    log_proto::Metric::scalar("load_ms", load_ms),
                    log_proto::Metric::scalar("staging_ms", staging_ms),
                    log_proto::Metric::scalar("train_ms", train_ms),
                ];
                log.write(&event, unix_time_ns()?)?;
                log.flush()?;
            }
        }
        if optimizer_step
            && completed != expected_completed
            && args
                .checkpoint_every
                .is_some_and(|interval| completed % interval == 0)
        {
            let base_output = args
                .training_output
                .as_ref()
                .expect("clap requires training output for periodic checkpoints");
            let output = periodic_checkpoint_path(base_output, completed);
            let metadata = training_metadata(
                &args,
                total_steps,
                base_learning_rate,
                warmup_steps,
                no_augmentation_steps,
                minimum_rate_steps,
                completed,
            );
            let current = variants
                .get(&current_size)
                .expect("current training size was compiled");
            save_training_state_new(&output, &current.model, &training_session, Some(&metadata))?;
            println!("checkpoint_step={completed} output={}", output.display());
            if let Some(log) = &mut event_log {
                log.write(
                    &log_proto::CheckpointPublished {
                        step: completed as u64,
                        uri: output.display().to_string(),
                        digest: None,
                        byte_length: output.metadata()?.len(),
                    },
                    unix_time_ns()?,
                )?;
                log.flush()?;
            }
        }
    }
    drop(profiler);
    pipeline.finish()?;
    if completed_micro_steps != expected_micro_steps || completed != expected_completed {
        return Err(
            format!(
                "training stopped at optimizer step {completed} / microstep {completed_micro_steps}, expected {expected_completed} / {expected_micro_steps}"
            )
            .into(),
        );
    }
    let run_seconds = run_started.elapsed().as_secs_f64();
    if let Some(output) = &args.output {
        let metadata = BTreeMap::from([
            ("format".to_owned(), "rxla-yolox-v1".to_owned()),
            ("steps".to_owned(), completed.to_string()),
        ])
        .into_iter()
        .collect();
        let current = variants
            .get(&current_size)
            .expect("current training size was compiled");
        if let Some(model_ema) = &current.ema {
            // EMA is packed into one device state to keep the hot step to one
            // bandwidth-sized update. Checkpointing is cold-path work, so split
            // it only once here before writing canonical parameter tensors.
            let packed = model_ema
                .packed_buffer(&training_session)?
                .to_vec::<f32>()?;
            let mut ema_buffers =
                Vec::with_capacity(model_ema.parameters().len() + model_ema.states().len());
            for parameter in model_ema.parameters() {
                ema_buffers.push((
                    parameter.path().to_owned(),
                    client.buffer(parameter.shape(), &packed[parameter.range()])?,
                ));
            }
            for state in model_ema.states() {
                ema_buffers.push((
                    state.path().to_owned(),
                    client.buffer(state.shape(), &packed[state.range()])?,
                ));
            }
            let mut buffers = ema_buffers
                .iter()
                .map(|(path, buffer)| (path.as_str(), buffer))
                .collect::<Vec<_>>();
            let states = current
                .model
                .state_buffers(&training_session)?
                .into_iter()
                .collect::<HashMap<_, _>>();
            for spec in current.model.schema().states() {
                if model_ema
                    .states()
                    .iter()
                    .any(|state| state.path() == spec.path())
                {
                    continue;
                }
                let buffer = states
                    .get(spec.path())
                    .copied()
                    .ok_or_else(|| format!("missing model state {:?}", spec.path()))?;
                buffers.push((spec.path(), buffer));
            }
            save_buffers_new(output, &buffers, Some(&metadata))?;
        } else {
            save_model_new(output, &current.model, &training_session, Some(&metadata))?;
        }
    }
    if let Some(output) = &args.training_output {
        let metadata = training_metadata(
            &args,
            total_steps,
            base_learning_rate,
            warmup_steps,
            no_augmentation_steps,
            minimum_rate_steps,
            completed,
        );
        let current = variants
            .get(&current_size)
            .expect("current training size was compiled");
        save_training_state_new(output, &current.model, &training_session, Some(&metadata))?;
    }
    let cache = compiler.stats();
    let executed_micro_steps = completed_micro_steps - replay_micro_steps;
    let executed_f64 = executed_micro_steps as f64;
    println!(
        "completed={completed} total_steps={total_steps} resumed_at={resumed_at} microsteps={completed_micro_steps} gradient_accumulation={} images={} augmentation_workers={} prefetch={} maximum_targets_per_image={maximum_targets_per_image} dropped_targets={dropped_targets} run_ms={:.1} images_per_second={:.1} mean_wait_ms={:.3} mean_load_ms={:.3} mean_staging_ms={:.3} mean_image_upload_ms={:.3} mean_train_ms={:.3} replay_ms={replay_ms:.1} train_compile_ms={train_compile_ms:.1} compiled_sizes={} compile_misses={} disk_cache_hits={}",
        args.gradient_accumulation,
        executed_micro_steps * args.batch_size,
        args.augmentation_workers,
        args.prefetch,
        run_seconds * 1e3,
        executed_micro_steps as f64 * args.batch_size as f64 / run_seconds,
        total_wait_ms / executed_f64,
        total_load_ms / executed_f64,
        total_staging_ms / executed_f64,
        total_image_upload_ms / executed_f64,
        total_train_ms / executed_f64,
        variants.len(),
        cache.misses,
        cache.disk_hits,
    );
    if let Some(mut log) = event_log {
        log.write(
            &log_proto::RunFinished {
                run_id,
                final_step: completed as u64,
                succeeded: true,
                message: None,
            },
            unix_time_ns()?,
        )?;
        log.finish()?;
    }
    Ok(())
}

impl Iterator for SourceSamples {
    type Item = std::result::Result<DetectionSample, String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next_sample()
            .transpose()
            .map(|result| result.map_err(|error| error.to_string()))
    }
}

impl<I> Iterator for PlannedBatches<I>
where
    I: Iterator<Item = std::result::Result<DetectionSample, String>>,
{
    type Item = std::result::Result<AugmentationBatch, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let mut groups = Vec::with_capacity(self.batch_size);
        let mut consumed = 0;
        for batch_index in 0..self.batch_size {
            let sample_rng = self
                .rng
                .sample(0, (self.step * self.batch_size + batch_index) as u64);
            let mosaic = self.step < self.mosaic_steps
                && sample_rng.bernoulli(0, 0, self.mosaic_probability);
            let count = if mosaic { 4 } else { 1 };
            let mut samples = Vec::with_capacity(count);
            for _ in 0..count {
                match take_training_sample(&mut self.samples, &mut consumed) {
                    Ok(sample) => samples.push(sample),
                    Err(error) => {
                        self.remaining = 0;
                        return Some(Err(error));
                    }
                }
            }
            let use_mixup = mosaic
                && samples.iter().any(|sample| !sample.boxes.is_empty())
                && sample_rng.bernoulli(0, 1, self.mixup_probability);
            let mixup = if use_mixup {
                loop {
                    match take_training_sample(&mut self.samples, &mut consumed) {
                        Ok(sample) if sample.boxes.is_empty() => continue,
                        Ok(sample) => break Some(sample),
                        Err(error) => {
                            self.remaining = 0;
                            return Some(Err(error));
                        }
                    }
                }
            } else {
                None
            };
            groups.push(AugmentationGroup {
                samples,
                mixup,
                rng: sample_rng,
            });
        }
        let size = self.sizes[self.step];
        self.remaining -= 1;
        self.step += 1;
        Some(Ok(AugmentationBatch { groups, size }))
    }
}

fn prepare_batch(
    groups: Vec<AugmentationGroup>,
    augmentation_size: u32,
    model_size: u32,
) -> std::result::Result<PreparedBatch, String> {
    let load_started = Instant::now();
    let prepared = groups
        .into_par_iter()
        .enumerate()
        .map(|(batch_index, group)| {
            let AugmentationGroup {
                samples,
                mixup,
                rng,
            } = group;
            if samples.len() == 4 {
                prepare_mosaic(samples, mixup, augmentation_size, batch_index, rng)
            } else {
                let sample = samples
                    .into_iter()
                    .next()
                    .ok_or_else(|| "empty augmentation sample group".to_owned())?;
                prepare_single(sample, augmentation_size, batch_index, rng)
            }
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    let mut pixels =
        Vec::with_capacity(prepared.len() * (augmentation_size * augmentation_size * 3) as usize);
    let mut targets = Vec::new();
    let mut sample_ids = Vec::with_capacity(prepared.len());
    for (id, sample_pixels, sample_targets) in prepared {
        sample_ids.push(id);
        pixels.extend(sample_pixels);
        targets.extend(sample_targets);
    }
    Ok(PreparedBatch {
        sample_ids,
        pixels,
        targets,
        size: model_size,
        load_ms: load_started.elapsed().as_secs_f64() * 1e3,
    })
}

fn take_training_sample<I>(
    samples: &mut I,
    consumed: &mut usize,
) -> std::result::Result<DetectionSample, String>
where
    I: Iterator<Item = std::result::Result<DetectionSample, String>>,
{
    match samples.next() {
        Some(Ok(sample)) => {
            *consumed += 1;
            Ok(sample)
        }
        Some(Err(error)) => Err(error),
        None => Err(format!(
            "dataset ended after {consumed} source samples in the next batch"
        )),
    }
}

fn prepare_single(
    sample: DetectionSample,
    size: u32,
    batch: usize,
    rng: SampleRng,
) -> std::result::Result<(String, Vec<u8>, Vec<DetectionTarget>), String> {
    let image = decode_sample(&sample)?;
    let (pixels, targets) = finish_training_sample(
        image,
        sample.boxes.clone(),
        &sample.classes,
        size,
        batch,
        rng,
    )?;
    Ok((sample.id, pixels, targets))
}

fn prepare_mosaic(
    samples: Vec<DetectionSample>,
    mixup: Option<DetectionSample>,
    size: u32,
    batch: usize,
    rng: SampleRng,
) -> std::result::Result<(String, Vec<u8>, Vec<DetectionTarget>), String> {
    let canvas_size = size * 2;
    let center_x = (size as f32 * (0.5 + rng.uniform(1, 0))) as i32;
    let center_y = (size as f32 * (0.5 + rng.uniform(1, 1))) as i32;
    let mut canvas = RgbImage::from_pixel(canvas_size, canvas_size, Rgb([114; 3]));
    let mut boxes = Vec::new();
    let mut classes = Vec::new();
    let mut ids = Vec::with_capacity(5);

    // A batch has only `batch_size` outer Rayon jobs, but Mosaic decodes four
    // independent JPEGs per output image. Expose those jobs to the same pool so
    // a small batch can still use all host cores while the device consumes the
    // preceding batch. Indexed parallel iteration preserves quadrant order.
    let tiles = samples
        .into_par_iter()
        .map(|sample| {
            let image = decode_sample(&sample)?;
            let scale =
                (size as f32 / image.height() as f32).min(size as f32 / image.width() as f32);
            let width = (image.width() as f32 * scale) as u32;
            let height = (image.height() as f32 * scale) as u32;
            let resized = resize_rgb(&image, width, height)?;
            Ok((sample, resized, scale))
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;

    for (quadrant, (sample, resized, scale)) in tiles.into_iter().enumerate() {
        let width = resized.width();
        let height = resized.height();
        let (large, small) = mosaic_coordinates(
            quadrant,
            center_x,
            center_y,
            width as i32,
            height as i32,
            canvas_size as i32,
        );
        let [lx1, ly1, lx2, ly2] = large;
        let [sx1, sy1, _, _] = small;
        copy_rgb_region(
            &mut canvas,
            &resized,
            [lx1 as u32, ly1 as u32],
            [sx1 as u32, sy1 as u32],
            [(lx2 - lx1) as u32, (ly2 - ly1) as u32],
        );
        let pad_x = (lx1 - sx1) as f32;
        let pad_y = (ly1 - sy1) as f32;
        for (bbox, class) in sample.boxes.iter().zip(&sample.classes) {
            boxes.push([
                (bbox[0] * scale + pad_x).clamp(0.0, canvas_size as f32),
                (bbox[1] * scale + pad_y).clamp(0.0, canvas_size as f32),
                (bbox[2] * scale + pad_x).clamp(0.0, canvas_size as f32),
                (bbox[3] * scale + pad_y).clamp(0.0, canvas_size as f32),
            ]);
            classes.push(*class);
        }
        ids.push(sample.id);
    }

    let matrix = random_affine_matrix(size, rng);
    let mut image = warp_affine(&canvas, size, matrix);
    let transformed = boxes
        .iter()
        .copied()
        .map(|bbox| transform_box(bbox, matrix, size as f32))
        .collect::<Vec<_>>();
    let mut transformed = transformed;
    if let Some(mixup) = mixup {
        ids.push(format!("mixup:{}", mixup.id));
        apply_mixup(&mut image, &mut transformed, &mut classes, mixup, size, rng)?;
    }
    let (pixels, targets) = finish_training_sample(image, transformed, &classes, size, batch, rng)?;
    Ok((ids.join("+"), pixels, targets))
}

fn copy_rgb_region(
    destination: &mut RgbImage,
    source: &RgbImage,
    [destination_x, destination_y]: [u32; 2],
    [source_x, source_y]: [u32; 2],
    [width, height]: [u32; 2],
) {
    debug_assert!(destination_x + width <= destination.width());
    debug_assert!(destination_y + height <= destination.height());
    debug_assert!(source_x + width <= source.width());
    debug_assert!(source_y + height <= source.height());
    let destination_stride = destination.width() as usize * 3;
    let source_stride = source.width() as usize * 3;
    let copy_bytes = width as usize * 3;
    let destination_x = destination_x as usize * 3;
    let destination_y = destination_y as usize;
    let source_x = source_x as usize * 3;
    let source_y = source_y as usize;
    let source = source.as_raw();
    destination
        .as_mut()
        .chunks_exact_mut(destination_stride)
        .skip(destination_y)
        .zip(source.chunks_exact(source_stride).skip(source_y))
        .take(height as usize)
        .for_each(|(destination, source)| {
            destination[destination_x..destination_x + copy_bytes]
                .copy_from_slice(&source[source_x..source_x + copy_bytes]);
        });
}

fn apply_mixup(
    image: &mut RgbImage,
    boxes: &mut Vec<[f32; 4]>,
    classes: &mut Vec<usize>,
    sample: DetectionSample,
    size: u32,
    rng: SampleRng,
) -> std::result::Result<(), String> {
    let source = decode_sample(&sample)?;
    let base_scale =
        (size as f32 / source.height() as f32).min(size as f32 / source.width() as f32);
    let resized_width = (source.width() as f32 * base_scale) as u32;
    let resized_height = (source.height() as f32 * base_scale) as u32;
    let resized = resize_rgb(&source, resized_width, resized_height)?;
    let mut copy = RgbImage::from_pixel(size, size, Rgb([114; 3]));
    image::imageops::replace(&mut copy, &resized, 0, 0);

    let jitter = 0.5 + rng.uniform(5, 0);
    let width = (size as f32 * jitter) as u32;
    let height = (size as f32 * jitter) as u32;
    let mut copy = resize_rgb(&copy, width, height)?;
    let flipped = rng.uniform(5, 1) > 0.5;
    if flipped {
        image::imageops::flip_horizontal_in_place(&mut copy);
    }
    let x_range = width.saturating_sub(size);
    let y_range = height.saturating_sub(size);
    let x_offset = (rng.uniform(5, 2) * x_range as f32) as u32;
    let y_offset = (rng.uniform(5, 3) * y_range as f32) as u32;

    image
        .as_mut()
        .par_chunks_mut(size as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..size as usize {
                let offset = x * 3;
                let copy_x = x as u32 + x_offset;
                let copy_y = y as u32 + y_offset;
                let overlay = if copy_x < width && copy_y < height {
                    copy.get_pixel(copy_x, copy_y).0
                } else {
                    [0; 3]
                };
                for channel in 0..3 {
                    row[offset + channel] = ((u16::from(row[offset + channel])
                        + u16::from(overlay[channel]))
                        / 2) as u8;
                }
            }
        });

    let scale = base_scale * jitter;
    for (bbox, class) in sample.boxes.into_iter().zip(sample.classes) {
        let mut bbox = bbox.map(|coordinate| coordinate * scale);
        bbox[0] = bbox[0].clamp(0.0, width as f32);
        bbox[2] = bbox[2].clamp(0.0, width as f32);
        bbox[1] = bbox[1].clamp(0.0, height as f32);
        bbox[3] = bbox[3].clamp(0.0, height as f32);
        if flipped {
            [bbox[0], bbox[2]] = [width as f32 - bbox[2], width as f32 - bbox[0]];
        }
        bbox[0] = (bbox[0] - x_offset as f32).clamp(0.0, size as f32);
        bbox[2] = (bbox[2] - x_offset as f32).clamp(0.0, size as f32);
        bbox[1] = (bbox[1] - y_offset as f32).clamp(0.0, size as f32);
        bbox[3] = (bbox[3] - y_offset as f32).clamp(0.0, size as f32);
        boxes.push(bbox);
        classes.push(class);
    }
    Ok(())
}

fn decode_sample(sample: &DetectionSample) -> std::result::Result<RgbImage, String> {
    use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};

    let image = if sample.encoded_image.starts_with(&[0xff, 0xd8]) {
        // Keep the common JPEG path direct, but dispatch by content rather than
        // extension: real datasets may contain PNG payloads under `.jpg` names.
        let options = DecoderOptions::default()
            .set_strict_mode(false)
            .set_max_width(usize::MAX)
            .set_max_height(usize::MAX)
            .jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
            ZCursor::new(sample.encoded_image.as_slice()),
            options,
        );
        decoder
            .decode_headers()
            .map_err(|error| error.to_string())?;
        let (width, height) = decoder
            .dimensions()
            .ok_or_else(|| "JPEG dimensions are unavailable after decoding headers".to_owned())?;
        let pixels = decoder.decode().map_err(|error| error.to_string())?;
        RgbImage::from_raw(width as u32, height as u32, pixels)
            .ok_or_else(|| "decoded JPEG RGB buffer has invalid dimensions".to_owned())?
    } else {
        image::load_from_memory(&sample.encoded_image)
            .map_err(|error| error.to_string())?
            .into_rgb8()
    };
    if sample
        .dimensions
        .is_some_and(|dimensions| dimensions != [image.width(), image.height()])
    {
        return Err("image dimensions do not match Fory metadata".to_owned());
    }
    Ok(image)
}

fn detection_targets(
    boxes: &[[f32; 4]],
    classes: &[usize],
    ratio: f32,
    batch: usize,
) -> Vec<DetectionTarget> {
    boxes
        .iter()
        .zip(classes)
        .map(|([x0, y0, x1, y1], &class)| DetectionTarget {
            batch,
            class,
            cxcywh: [
                (x0 + x1) * 0.5 * ratio,
                (y0 + y1) * 0.5 * ratio,
                (x1 - x0) * ratio,
                (y1 - y0) * ratio,
            ],
        })
        .collect()
}

fn finish_training_sample(
    mut image: RgbImage,
    mut boxes: Vec<[f32; 4]>,
    classes: &[usize],
    size: u32,
    batch: usize,
    rng: SampleRng,
) -> std::result::Result<(Vec<u8>, Vec<DetectionTarget>), String> {
    let ratio = (size as f32 / image.width() as f32).min(size as f32 / image.height() as f32);
    let valid = boxes
        .iter()
        .map(|bbox| ((bbox[2] - bbox[0]) * ratio, (bbox[3] - bbox[1]) * ratio))
        .map(|(width, height)| width.min(height) > 1.0)
        .collect::<Vec<_>>();

    // Official TrainTransform falls back to the original, unaugmented image
    // and labels when augmentation would leave no box larger than one pixel.
    // The validity predicate is invariant under HSV and horizontal flip, so
    // avoid cloning/mutating the image when the fallback is already known.
    if !boxes.is_empty() && !valid.iter().any(|&keep| keep) {
        let (pixels, ratio) = letterbox(&image, size)?;
        return Ok((pixels, detection_targets(&boxes, classes, ratio, batch)));
    }
    if boxes.is_empty() {
        return Ok((letterbox(&image, size)?.0, Vec::new()));
    }

    train_transform(&mut image, &mut boxes, rng);
    let (pixels, ratio) = letterbox(&image, size)?;
    let targets = detection_targets(&boxes, classes, ratio, batch)
        .into_iter()
        .zip(valid)
        .filter_map(|(target, keep)| keep.then_some(target))
        .collect();
    Ok((pixels, targets))
}

fn train_transform(image: &mut RgbImage, boxes: &mut [[f32; 4]], rng: SampleRng) {
    augment_hsv(image, rng);
    if rng.bernoulli(4, 0, 0.5) {
        image::imageops::flip_horizontal_in_place(image);
        let width = image.width() as f32;
        for bbox in boxes {
            [bbox[0], bbox[2]] = [width - bbox[2], width - bbox[0]];
        }
    }
}

fn augment_hsv(image: &mut RgbImage, rng: SampleRng) {
    let enabled = [
        rng.bernoulli(3, 0, 0.5),
        rng.bernoulli(3, 1, 0.5),
        rng.bernoulli(3, 2, 0.5),
    ];
    let hue_delta = if enabled[0] {
        ((rng.uniform(3, 3) * 2.0 - 1.0) * 5.0).trunc()
    } else {
        0.0
    };
    let saturation_delta = if enabled[1] {
        ((rng.uniform(3, 4) * 2.0 - 1.0) * 30.0).trunc()
    } else {
        0.0
    };
    let value_delta = if enabled[2] {
        ((rng.uniform(3, 5) * 2.0 - 1.0) * 30.0).trunc()
    } else {
        0.0
    };
    if hue_delta == 0.0 && saturation_delta == 0.0 && value_delta == 0.0 {
        return;
    }
    image.as_mut().par_chunks_mut(3).for_each(|pixel| {
        pixel.copy_from_slice(&adjust_hsv_pixel(
            [pixel[0], pixel[1], pixel[2]],
            [hue_delta, saturation_delta, value_delta],
        ));
    });
}

fn adjust_hsv_pixel(
    pixel: [u8; 3],
    [hue_delta, saturation_delta, value_delta]: [f32; 3],
) -> [u8; 3] {
    let [mut hue, mut saturation, mut value] = rgb_to_opencv_hsv(pixel);
    hue += hue_delta;
    if hue < 0.0 {
        hue += 180.0;
    } else if hue >= 180.0 {
        hue -= 180.0;
    }
    saturation = (saturation + saturation_delta).clamp(0.0, 255.0);
    value = (value + value_delta).clamp(0.0, 255.0);
    opencv_hsv_to_rgb([hue, saturation, value])
}

const HSV_SHIFT: i32 = 12;
const HSV_ROUND: i32 = 1 << (HSV_SHIFT - 1);
const HSV_SATURATION_DIVISORS: [i32; 256] = hsv_divisors(255, 1);
const HSV_HUE_DIVISORS: [i32; 256] = hsv_divisors(180, 6);

const fn hsv_divisors(numerator: i32, denominator_scale: i32) -> [i32; 256] {
    let mut table = [0; 256];
    let mut value = 1;
    while value < 256 {
        let denominator = denominator_scale * value as i32;
        table[value] = ((numerator << HSV_SHIFT) + denominator / 2) / denominator;
        value += 1;
    }
    table
}

fn rgb_to_opencv_hsv([red, green, blue]: [u8; 3]) -> [f32; 3] {
    let red = i32::from(red);
    let green = i32::from(green);
    let blue = i32::from(blue);
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let saturation = (delta * HSV_SATURATION_DIVISORS[maximum as usize] + HSV_ROUND) >> HSV_SHIFT;
    let hue_base = if maximum == red {
        green - blue
    } else if maximum == green {
        blue - red + 2 * delta
    } else {
        red - green + 4 * delta
    };
    let mut hue = (hue_base * HSV_HUE_DIVISORS[delta as usize] + HSV_ROUND) >> HSV_SHIFT;
    if hue < 0 {
        hue += 180;
    }
    [hue as f32, saturation as f32, maximum as f32]
}

fn opencv_hsv_to_rgb([hue, saturation, value]: [f32; 3]) -> [u8; 3] {
    let chroma = value * saturation / 255.0;
    let section = hue / 30.0;
    let sector = section as usize;
    let fraction = section - sector as f32;
    let x = chroma
        * if sector.is_multiple_of(2) {
            fraction
        } else {
            1.0 - fraction
        };
    let (red, green, blue) = match sector {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let offset = value - chroma;
    [red, green, blue].map(|channel| (channel + offset + 0.5) as u8)
}

fn mosaic_coordinates(
    quadrant: usize,
    center_x: i32,
    center_y: i32,
    width: i32,
    height: i32,
    canvas: i32,
) -> ([i32; 4], [i32; 4]) {
    match quadrant {
        0 => {
            let large = [
                (center_x - width).max(0),
                (center_y - height).max(0),
                center_x,
                center_y,
            ];
            let small = [
                width - (large[2] - large[0]),
                height - (large[3] - large[1]),
                width,
                height,
            ];
            (large, small)
        }
        1 => {
            let large = [
                center_x,
                (center_y - height).max(0),
                (center_x + width).min(canvas),
                center_y,
            ];
            let small = [
                0,
                height - (large[3] - large[1]),
                (large[2] - large[0]).min(width),
                height,
            ];
            (large, small)
        }
        2 => {
            let large = [
                (center_x - width).max(0),
                center_y,
                center_x,
                (center_y + height).min(canvas),
            ];
            let small = [
                width - (large[2] - large[0]),
                0,
                width,
                (large[3] - large[1]).min(height),
            ];
            (large, small)
        }
        _ => {
            let large = [
                center_x,
                center_y,
                (center_x + width).min(canvas),
                (center_y + height).min(canvas),
            ];
            let small = [
                0,
                0,
                (large[2] - large[0]).min(width),
                (large[3] - large[1]).min(height),
            ];
            (large, small)
        }
    }
}

fn random_affine_matrix(size: u32, rng: SampleRng) -> [[f32; 3]; 2] {
    let angle = (-10.0 + 20.0 * rng.uniform(2, 0)).to_radians();
    // YOLOX-tiny overrides the base recipe's 0.1..2.0 range with 0.5..1.5.
    let scale = 0.5 + rng.uniform(2, 1);
    let shear_x = (-2.0 + 4.0 * rng.uniform(2, 2)).to_radians().tan();
    let shear_y = (-2.0 + 4.0 * rng.uniform(2, 3)).to_radians().tan();
    let alpha = scale * angle.cos();
    let beta = scale * angle.sin();
    let extent = size as f32;
    [
        [
            alpha - shear_y * beta,
            beta + shear_y * alpha,
            (-0.1 + 0.2 * rng.uniform(2, 4)) * extent,
        ],
        [
            -beta + shear_x * alpha,
            alpha + shear_x * beta,
            (-0.1 + 0.2 * rng.uniform(2, 5)) * extent,
        ],
    ]
}

fn transform_box(bbox: [f32; 4], matrix: [[f32; 3]; 2], extent: f32) -> [f32; 4] {
    let corners = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[1]],
    ]
    .map(|[x, y]| {
        [
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
        ]
    });
    let xs = corners.map(|point| point[0]);
    let ys = corners.map(|point| point[1]);
    [
        xs.iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .clamp(0.0, extent),
        ys.iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .clamp(0.0, extent),
        xs.iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .clamp(0.0, extent),
        ys.iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .clamp(0.0, extent),
    ]
}

fn interpolate_rgb_bilinear(image: &RgbImage, x: f32, y: f32, border: u8) -> [u8; 3] {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let bytes = image.as_raw();
    let sample = |sample_x: i32, sample_y: i32| {
        if sample_x < 0 || sample_y < 0 || sample_x >= width || sample_y >= height {
            [f32::from(border); 3]
        } else {
            let offset = ((sample_y * width + sample_x) * 3) as usize;
            [
                f32::from(bytes[offset]),
                f32::from(bytes[offset + 1]),
                f32::from(bytes[offset + 2]),
            ]
        }
    };
    let top_left = sample(x0, y0);
    let top_right = sample(x1, y0);
    let bottom_left = sample(x0, y1);
    let bottom_right = sample(x1, y1);
    std::array::from_fn(|channel| {
        let top = top_left[channel] + (top_right[channel] - top_left[channel]) * fx;
        let bottom = bottom_left[channel] + (bottom_right[channel] - bottom_left[channel]) * fx;
        (top + (bottom - top) * fy + 0.5) as u8
    })
}

fn warp_affine(image: &RgbImage, size: u32, matrix: [[f32; 3]; 2]) -> RgbImage {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let inverse = [
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ];
    let mut output = vec![114; (size * size * 3) as usize];
    output
        .par_chunks_mut(size as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let translated_y = y as f32 - matrix[1][2];
            let source_x_from_y = inverse[0][1] * translated_y;
            let source_y_from_y = inverse[1][1] * translated_y;
            for x in 0..size as usize {
                let translated_x = x as f32 - matrix[0][2];
                let source_x = inverse[0][0] * translated_x + source_x_from_y;
                let source_y = inverse[1][0] * translated_x + source_y_from_y;
                let pixel = interpolate_rgb_bilinear(image, source_x, source_y, 114);
                row[x * 3..x * 3 + 3].copy_from_slice(&pixel);
            }
        });
    RgbImage::from_raw(size, size, output).expect("affine output has exact RGB dimensions")
}

impl TrainingSource {
    fn open(source: DataSource) -> Result<Self> {
        match source {
            DataSource::Coco {
                annotations,
                images,
                image_id,
            } => {
                let document: Coco = serde_json::from_reader(std::io::BufReader::new(
                    std::fs::File::open(annotations)?,
                ))?;
                let position = document
                    .images
                    .iter()
                    .position(|image| image.id == image_id)
                    .ok_or("image ID is absent from annotations")?;
                let categories = category_map(document.categories.iter().map(|value| value.id));
                let mut annotations = HashMap::<u64, Vec<usize>>::new();
                for (index, annotation) in document.annotations.iter().enumerate() {
                    if annotation.iscrowd == 0 {
                        annotations
                            .entry(annotation.image_id)
                            .or_default()
                            .push(index);
                    }
                }
                Ok(Self::Coco {
                    document,
                    images,
                    position,
                    rollover_position: 0,
                    annotations,
                    categories,
                })
            }
            DataSource::Tfrecord { input } => {
                let manifest_path = PathBuf::from(format!("{}.json", input.display()));
                let manifest: ForyManifest = serde_json::from_reader(std::io::BufReader::new(
                    std::fs::File::open(&manifest_path)?,
                ))?;
                if manifest.codec != "apache-fory-standard-row-1.7"
                    || manifest.container != "tfrecord"
                    || manifest.annotation_policy != "yolox-coco-v1"
                    || manifest.schema_sha256 != DETECTION_SCHEMA_SHA256
                {
                    return Err(format!(
                        "incompatible TFRecord/Fory manifest {}",
                        manifest_path.display()
                    )
                    .into());
                }
                Ok(Self::Tfrecord {
                    reader: TfrecordReader::open(&input)?,
                    input,
                    categories: category_map(COCO_CATEGORY_IDS),
                    samples: manifest.samples,
                })
            }
        }
    }

    fn sample_count(&self) -> usize {
        match self {
            Self::Coco { document, .. } => document.images.len(),
            Self::Tfrecord { samples, .. } => *samples,
        }
    }

    fn next_sample(&mut self) -> Result<Option<DetectionSample>> {
        match self {
            Self::Coco {
                document,
                images,
                position,
                rollover_position,
                annotations,
                categories,
            } => {
                if document.images.get(*position).is_none() {
                    *position = *rollover_position;
                }
                let image = document
                    .images
                    .get(*position)
                    .ok_or("COCO dataset contains no image at its start position")?;
                *position += 1;
                let mut boxes = Vec::new();
                let mut classes = Vec::new();
                for &index in annotations.get(&image.id).into_iter().flatten() {
                    let annotation = &document.annotations[index];
                    let Some(bbox) = clean_coco_bbox(
                        annotation.bbox,
                        annotation.area,
                        [image.width, image.height],
                    ) else {
                        continue;
                    };
                    boxes.push(bbox);
                    classes.push(
                        *categories
                            .get(&annotation.category_id)
                            .ok_or("annotation references an unknown category")?,
                    );
                }
                Ok(Some(DetectionSample {
                    id: image.id.to_string(),
                    encoded_image: std::fs::read(images.join(&image.file_name))?,
                    dimensions: None,
                    boxes,
                    classes,
                }))
            }
            Self::Tfrecord {
                reader,
                input,
                categories,
                ..
            } => {
                let payload = if let Some(payload) = reader.read_record()? {
                    payload
                } else {
                    *reader = TfrecordReader::open(&*input)?;
                    let Some(payload) = reader.read_record()? else {
                        return Ok(None);
                    };
                    payload
                };
                let row = from_row::<DetectionRecord>(payload)?;
                let boxes = row.boxes()?;
                let labels = row.labels()?;
                if boxes.len() != labels.len() * 4 {
                    return Err("Fory detection row has inconsistent boxes and labels".into());
                }
                let dimensions = [u32::try_from(row.width()?)?, u32::try_from(row.height()?)?];
                let decoded_boxes = (0..labels.len())
                    .map(|index| {
                        Ok([
                            boxes.get(index * 4)?,
                            boxes.get(index * 4 + 1)?,
                            boxes.get(index * 4 + 2)?,
                            boxes.get(index * 4 + 3)?,
                        ])
                    })
                    .collect::<std::result::Result<Vec<_>, fory::Error>>()?;
                let decoded_classes = labels
                    .iter()
                    .map(|label| {
                        let label = u64::try_from(label.map_err(|error| error.to_string())?)
                            .map_err(|error| error.to_string())?;
                        categories
                            .get(&label)
                            .copied()
                            .ok_or_else(|| format!("unknown COCO category {label}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let mut boxes = Vec::with_capacity(decoded_boxes.len());
                let mut classes = Vec::with_capacity(decoded_classes.len());
                for (bbox, class) in decoded_boxes.into_iter().zip(decoded_classes) {
                    if let Some(bbox) = clean_xyxy_bbox(bbox, dimensions) {
                        boxes.push(bbox);
                        classes.push(class);
                    }
                }
                Ok(Some(DetectionSample {
                    id: row.sample_id()?.to_owned(),
                    encoded_image: row.image()?.to_vec(),
                    dimensions: Some(dimensions),
                    boxes,
                    classes,
                }))
            }
        }
    }
}

fn clean_coco_bbox(
    [x, y, width, height]: [f32; 4],
    area: f32,
    [image_width, image_height]: [u32; 2],
) -> Option<[f32; 4]> {
    if ![x, y, width, height, area].into_iter().all(f32::is_finite) || area <= 0.0 {
        return None;
    }
    let x1 = x.max(0.0);
    let y1 = y.max(0.0);
    clean_xyxy_bbox(
        [x1, y1, x1 + width.max(0.0), y1 + height.max(0.0)],
        [image_width, image_height],
    )
}

fn clean_xyxy_bbox(
    [x1, y1, x2, y2]: [f32; 4],
    [image_width, image_height]: [u32; 2],
) -> Option<[f32; 4]> {
    if ![x1, y1, x2, y2].into_iter().all(f32::is_finite) {
        return None;
    }
    let bbox = [
        x1.max(0.0),
        y1.max(0.0),
        x2.min(image_width as f32),
        y2.min(image_height as f32),
    ];
    (bbox[2] >= bbox[0] && bbox[3] >= bbox[1]).then_some(bbox)
}

fn category_map(ids: impl IntoIterator<Item = u64>) -> BTreeMap<u64, usize> {
    let mut ids = ids.into_iter().collect::<Vec<_>>();
    ids.sort_unstable();
    ids.into_iter()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect()
}

const COCO_CATEGORY_IDS: [u64; 80] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
];

const DETECTION_SCHEMA_SHA256: &str =
    "237f2fa5b001dbed66bc326f618b90fdeba67d2d284357a38274612d007d455f";

fn padded_targets(
    batch: usize,
    max_targets: usize,
    classes: usize,
    targets: &[DetectionTarget],
) -> Result<PaddedTargets> {
    let mut boxes = vec![0.0; batch * max_targets * 4];
    let mut class_indices = vec![0; batch * max_targets];
    let mut valid = vec![0.0; batch * max_targets];
    let mut counts = vec![0; batch];
    for target in targets {
        if target.batch >= batch || target.class >= classes {
            return Err("target batch or class is out of range".into());
        }
        let slot = counts[target.batch];
        counts[target.batch] += 1;
        if slot >= max_targets {
            continue;
        }
        let row = target.batch * max_targets + slot;
        boxes[row * 4..row * 4 + 4].copy_from_slice(&target.cxcywh);
        class_indices[row] = i32::try_from(target.class)?;
        valid[row] = 1.0;
    }
    let maximum_per_image = counts.iter().copied().max().unwrap_or(0);
    let dropped = counts
        .into_iter()
        .map(|count| count.saturating_sub(max_targets))
        .sum();
    Ok(PaddedTargets {
        boxes,
        classes: class_indices,
        valid,
        stats: TargetPaddingStats {
            maximum_per_image,
            dropped,
        },
    })
}

fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> std::result::Result<RgbImage, String> {
    let source = ImageRef::new(
        image.width(),
        image.height(),
        image.as_raw(),
        PixelType::U8x3,
    )
    .map_err(|error| error.to_string())?;
    let mut destination = FastImage::new(width, height, PixelType::U8x3);
    let options =
        ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FastFilterType::Bilinear));
    Resizer::new()
        .resize(&source, &mut destination, &options)
        .map_err(|error| error.to_string())?;
    RgbImage::from_raw(width, height, destination.into_vec())
        .ok_or_else(|| "resized RGB buffer has invalid dimensions".to_owned())
}

fn letterbox(image: &RgbImage, size: u32) -> std::result::Result<(Vec<u8>, f32), String> {
    let ratio = (size as f32 / image.width() as f32).min(size as f32 / image.height() as f32);
    let width = (image.width() as f32 * ratio) as u32;
    let height = (image.height() as f32 * ratio) as u32;
    let resized = resize_rgb(image, width, height)?;
    let mut output = vec![114; (size * size * 3) as usize];
    output
        .par_chunks_mut(size as usize * 3)
        .take(height as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                row[x * 3..x * 3 + 3].copy_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            }
        });
    Ok((output, ratio))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_to_the_official_tiny_input_recipe() {
        let args = Args::try_parse_from([
            "yolox_coco_train_step",
            "plugin.so",
            "weights.safetensors",
            "coco",
            "instances.json",
            "images",
            "1",
        ])
        .unwrap();
        assert_eq!(args.size, 416);
        assert!(args.random_resize);
        assert_eq!((args.random_size_min, args.random_size_max), (10, 20));
        assert_eq!(args.mosaic_probability, 1.0);
        assert_eq!(args.mixup_probability, 0.0);
    }

    #[test]
    fn periodic_checkpoints_are_immutable_versioned_siblings() {
        assert_eq!(
            periodic_checkpoint_path(Path::new("runs/train.state.safetensors"), 12_345),
            PathBuf::from("runs/train.state.step-12345.safetensors")
        );
        assert_eq!(
            periodic_checkpoint_path(Path::new("runs/train-state"), 7),
            PathBuf::from("runs/train-state.step-7")
        );
    }

    #[test]
    fn official_epoch_recipe_keeps_data_and_lr_boundaries_distinct() {
        assert_eq!(
            official_epoch_schedule(118_287, 64, 300),
            Some((554_700, 9_245, 29_584, 27_735))
        );
        assert_eq!(official_epoch_schedule(0, 64, 300), None);
        assert_eq!(official_epoch_schedule(118_287, 0, 300), None);
    }

    #[test]
    fn coco_boxes_follow_the_official_cleaning_policy() {
        assert_eq!(
            clean_coco_bbox([-5.0, 3.0, 12.0, 20.0], 240.0, [10, 15]),
            Some([0.0, 3.0, 10.0, 15.0])
        );
        assert_eq!(
            clean_coco_bbox([1.0, 2.0, -3.0, 4.0], 1.0, [10, 10]),
            Some([1.0, 2.0, 1.0, 6.0])
        );
        assert_eq!(clean_coco_bbox([1.0; 4], 0.0, [10, 10]), None);
        assert_eq!(clean_coco_bbox([12.0, 1.0, 3.0, 3.0], 9.0, [10, 10]), None);
        assert_eq!(clean_xyxy_bbox([0.0, 0.0, f32::NAN, 1.0], [10, 10]), None);
    }

    #[test]
    fn sample_decoder_dispatches_non_jpeg_payloads_by_content() {
        let source = RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mut encoded = std::io::Cursor::new(Vec::new());
        source
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let sample = DetectionSample {
            id: "png-under-any-name".to_owned(),
            encoded_image: encoded.into_inner(),
            dimensions: Some([2, 1]),
            boxes: Vec::new(),
            classes: Vec::new(),
        };

        assert_eq!(decode_sample(&sample).unwrap(), source);
    }

    #[test]
    fn targets_are_padded_as_class_indices_and_capped_per_image() {
        let targets = [
            DetectionTarget {
                batch: 1,
                class: 2,
                cxcywh: [1.0, 2.0, 3.0, 4.0],
            },
            DetectionTarget {
                batch: 1,
                class: 1,
                cxcywh: [5.0, 6.0, 7.0, 8.0],
            },
        ];
        let padded = padded_targets(2, 1, 3, &targets).unwrap();
        assert_eq!(&padded.boxes[4..8], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(padded.classes, [0, 2]);
        assert_eq!(padded.valid, [0.0, 1.0]);
        assert_eq!(
            padded.stats,
            TargetPaddingStats {
                maximum_per_image: 2,
                dropped: 1,
            }
        );
    }

    #[test]
    fn train_transform_falls_back_when_every_box_is_tiny() {
        let image = RgbImage::from_pixel(2, 2, Rgb([10, 20, 30]));
        let (pixels, targets) = finish_training_sample(
            image,
            vec![[0.0, 0.0, 0.5, 0.5]],
            &[7],
            2,
            0,
            DataRng::new(3).sample(0, 0),
        )
        .unwrap();
        assert_eq!(pixels, [30, 20, 10].repeat(4));
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].class, 7);
        assert_eq!(targets[0].cxcywh, [0.25, 0.25, 0.5, 0.5]);
    }

    #[test]
    fn train_transform_filters_tiny_boxes_when_another_box_survives() {
        let image = RgbImage::from_pixel(8, 8, Rgb([10, 20, 30]));
        let (_, targets) = finish_training_sample(
            image,
            vec![[0.0, 0.0, 0.5, 0.5], [1.0, 1.0, 5.0, 5.0]],
            &[7, 8],
            8,
            0,
            DataRng::new(3).sample(0, 0),
        )
        .unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].class, 8);
    }

    #[test]
    fn mosaic_regions_have_matching_valid_source_and_destination_extents() {
        for quadrant in 0..4 {
            for center_x in [208, 416, 624] {
                for center_y in [208, 416, 624] {
                    let (large, small) =
                        mosaic_coordinates(quadrant, center_x, center_y, 333, 277, 832);
                    assert_eq!(large[2] - large[0], small[2] - small[0]);
                    assert_eq!(large[3] - large[1], small[3] - small[1]);
                    assert!(large[0] >= 0 && large[1] >= 0);
                    assert!(large[2] <= 832 && large[3] <= 832);
                    assert!(small[0] >= 0 && small[1] >= 0);
                    assert!(small[2] <= 333 && small[3] <= 277);
                }
            }
        }
    }

    #[test]
    fn rgb_region_copy_writes_only_the_selected_rectangle() {
        let source = RgbImage::from_fn(3, 2, |x, y| {
            let value = (y * 3 + x) as u8;
            Rgb([value, value + 10, value + 20])
        });
        let mut destination = RgbImage::from_pixel(4, 3, Rgb([99; 3]));
        copy_rgb_region(&mut destination, &source, [1, 1], [1, 0], [2, 2]);
        assert_eq!(destination.get_pixel(1, 1), source.get_pixel(1, 0));
        assert_eq!(destination.get_pixel(2, 1), source.get_pixel(2, 0));
        assert_eq!(destination.get_pixel(1, 2), source.get_pixel(1, 1));
        assert_eq!(destination.get_pixel(2, 2), source.get_pixel(2, 1));
        assert_eq!(*destination.get_pixel(0, 0), Rgb([99; 3]));
        assert_eq!(*destination.get_pixel(3, 2), Rgb([99; 3]));
    }

    #[test]
    fn identity_affine_preserves_boxes_and_pixels() {
        let matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(
            transform_box([1.0, 2.0, 5.0, 7.0], matrix, 8.0),
            [1.0, 2.0, 5.0, 7.0]
        );
        let image = RgbImage::from_fn(8, 8, |x, y| Rgb([x as u8, y as u8, 3]));
        assert_eq!(warp_affine(&image, 8, matrix), image);
    }

    #[test]
    fn specialized_bilinear_matches_image_reference() {
        let image = RgbImage::from_fn(7, 5, |x, y| {
            Rgb([
                (x * 31 + y * 7) as u8,
                (x * 3 + y * 41) as u8,
                (x * 17 + y * 13) as u8,
            ])
        });
        for y in [0.0, 0.2, 1.75, 4.0] {
            for x in [0.0, 0.6, 3.125, 6.0] {
                let expected = image::imageops::interpolate_bilinear(&image, x, y).unwrap();
                assert_eq!(interpolate_rgb_bilinear(&image, x, y, 114), expected.0);
            }
        }
    }

    #[test]
    fn affine_warp_tracks_opencv_reference() {
        let pixels = (0..48).map(|value| (value * 5) as u8).collect();
        let image = RgbImage::from_raw(4, 4, pixels).unwrap();
        let actual = warp_affine(&image, 4, [[0.9, 0.1, 0.3], [-0.05, 1.05, -0.2]]).into_raw();
        let expected = [
            47, 50, 53, 25, 30, 35, 45, 50, 55, 64, 69, 74, 89, 91, 94, 80, 85, 90, 100, 105, 110,
            120, 125, 130, 118, 121, 123, 135, 140, 145, 155, 160, 165, 175, 180, 185, 136, 137,
            139, 182, 186, 191, 192, 197, 201, 201, 205, 210,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn simd_resize_matches_opencv_linear_within_one_lsb() {
        let pixels = (0..36).map(|value| (value * 7) as u8).collect();
        let image = RgbImage::from_raw(4, 3, pixels).unwrap();
        let resized = resize_rgb(&image, 3, 5).unwrap();
        let opencv = [
            3, 10, 17, 31, 38, 45, 59, 66, 73, 37, 44, 51, 65, 72, 79, 93, 100, 107, 87, 94, 101,
            116, 123, 130, 144, 151, 158, 138, 145, 152, 166, 173, 180, 194, 201, 208, 171, 178,
            185, 199, 206, 213, 227, 234, 241,
        ];
        assert!(
            resized
                .into_raw()
                .into_iter()
                .zip(opencv)
                .all(|(actual, expected)| actual.abs_diff(expected) <= 1)
        );
    }

    #[test]
    fn hsv_conversion_round_trips_primary_colors() {
        for color in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]] {
            assert_eq!(opencv_hsv_to_rgb(rgb_to_opencv_hsv(color)), color);
        }
    }

    #[test]
    fn hsv_adjustment_tracks_opencv_reference() {
        let input = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 64, 32],
            [17, 99, 201],
            [240, 241, 239],
            [0, 0, 0],
        ];
        let opencv = [
            [255, 49, 17],
            [17, 255, 49],
            [49, 17, 255],
            [151, 96, 48],
            [34, 91, 224],
            [255, 255, 255],
            [23, 23, 23],
        ];
        for (pixel, expected) in input.into_iter().zip(opencv) {
            let actual = adjust_hsv_pixel(pixel, [4.0, -17.0, 23.0]);
            assert!(
                actual
                    .into_iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.abs_diff(expected) <= 2),
                "actual={actual:?} expected={expected:?}",
            );
        }
    }

    #[test]
    fn default_learning_rate_scales_official_per_image_rate() {
        assert_eq!(default_learning_rate(8), 0.00125);
        assert_eq!(default_learning_rate(64), 0.01);
    }

    #[test]
    fn random_resize_keeps_ten_step_blocks_and_is_reproducible() {
        let first = training_sizes(416, 31, 7, true, 10, 20);
        let second = training_sizes(416, 31, 7, true, 10, 20);
        assert_eq!(first, second);
        assert!(first[..10].iter().all(|&size| size == 416));
        assert!(first[10..20].iter().all(|&size| size == first[10]));
        assert!(first[20..30].iter().all(|&size| size == first[20]));
        assert!(
            first[10..]
                .iter()
                .all(|&size| size % 32 == 0 && (320..=640).contains(&size))
        );
        assert_eq!(training_sizes(416, 31, 7, false, 10, 20), vec![416; 31]);
    }

    #[test]
    fn random_resize_keeps_host_augmentation_at_the_base_size() {
        let image = RgbImage::from_pixel(4, 2, Rgb([10, 20, 30]));
        let mut encoded_image = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut encoded_image)
            .encode_image(&image)
            .unwrap();
        let prepared = prepare_batch(
            vec![AugmentationGroup {
                samples: vec![DetectionSample {
                    id: "sample".to_owned(),
                    encoded_image,
                    dimensions: Some([4, 2]),
                    boxes: Vec::new(),
                    classes: Vec::new(),
                }],
                mixup: None,
                rng: DataRng::new(3).sample(0, 0),
            }],
            8,
            12,
        )
        .unwrap();
        assert_eq!(prepared.pixels.len(), 8 * 8 * 3);
        assert_eq!(prepared.size, 12);
    }

    #[test]
    fn coco_source_rolls_over_without_ending_the_training_stream() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("one.jpg"), [1]).unwrap();
        std::fs::write(directory.path().join("two.jpg"), [2]).unwrap();
        let mut source = TrainingSource::Coco {
            document: Coco {
                images: vec![
                    CocoImage {
                        id: 1,
                        file_name: "one.jpg".to_owned(),
                        width: 1,
                        height: 1,
                    },
                    CocoImage {
                        id: 2,
                        file_name: "two.jpg".to_owned(),
                        width: 1,
                        height: 1,
                    },
                ],
                annotations: Vec::new(),
                categories: Vec::new(),
            },
            images: directory.path().to_owned(),
            position: 0,
            rollover_position: 0,
            annotations: HashMap::new(),
            categories: BTreeMap::new(),
        };

        let ids = (0..5)
            .map(|_| source.next_sample().unwrap().unwrap().id)
            .collect::<Vec<_>>();
        assert_eq!(ids, ["1", "2", "1", "2", "1"]);
    }
}
