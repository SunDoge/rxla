//! Train YOLOX on real COCO samples with device SimOTA and resident momentum SGD.

use clap::{Parser, Subcommand, ValueEnum};
use fory::{ForyRow, from_row};
use image::{Rgb, RgbImage, imageops::FilterType};
use rayon::prelude::*;
use rxla_core::{CacheLimits, Client, Compiler, DType, Tensor};
use rxla_models::{
    YoloConfig, YoloLossConfig, YoloSimOtaConfig, YoloTargets, yolo, yolox_loss_with_l1_weight,
    yolox_simota,
};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::{SafeTensors, save_model_new};
use rxla_tfrecord::Reader as TfrecordReader;
use rxla_train::{
    BoundedPipeline, DataRng, DetectionTarget, SampleRng, SgdOptions, ShufflePool, YoloLrSchedule,
    apply_model_sgd_with_learning_rate,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MAX_TARGETS: usize = 120;

fn default_augmentation_workers() -> usize {
    std::thread::available_parallelism()
        .map(|threads| (threads.get() * 2 / 3).max(1))
        .unwrap_or(1)
}

#[derive(Parser)]
#[command(about = "Run consecutive real COCO YOLOX training steps")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Checkpoint produced by benchmarks/convert_yolox.py.
    weights: PathBuf,
    #[command(subcommand)]
    source: DataSource,
    /// Number of consecutive annotated images to train on.
    #[arg(long, default_value_t = 1)]
    steps: usize,
    #[arg(long, default_value_t = 64)]
    size: i64,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Number of prepared host batches retained ahead of the device.
    #[arg(long, default_value_t = default_augmentation_workers())]
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
    #[arg(long, default_value_t = 0.00015625)]
    learning_rate: f32,
    /// Internal CUDA arithmetic; model inputs, parameters, state, and outputs remain F32.
    #[arg(long, value_enum, default_value_t = ComputeDtype::F32)]
    compute_dtype: ComputeDtype,
    /// Override warmup duration (default: 5/300 of all steps).
    #[arg(long)]
    warmup_steps: Option<usize>,
    /// Override the minimum-rate tail (default: 15/300 of all steps).
    #[arg(long)]
    no_augmentation_steps: Option<usize>,
    /// Write inference-ready trained parameters and BatchNorm state here.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Print one progress line every N completed steps.
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

#[derive(Subcommand)]
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
}

#[derive(Deserialize)]
struct CocoAnnotation {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
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
    schema_sha256: String,
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
    load_ms: f64,
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
    size: u32,
    rng: DataRng,
}

enum TrainingSource {
    Coco {
        document: Coco,
        images: PathBuf,
        position: usize,
        annotations: HashMap<u64, Vec<usize>>,
        categories: BTreeMap<u64, usize>,
    },
    Tfrecord {
        reader: TfrecordReader<std::io::BufReader<std::fs::File>>,
        categories: BTreeMap<u64, usize>,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size <= 0
        || args.size % 32 != 0
        || !args.learning_rate.is_finite()
        || args.steps == 0
        || args.log_every == 0
        || args.batch_size == 0
        || args.prefetch == 0
        || args.augmentation_workers == 0
        || args.shuffle_buffer == 0
        || !(0.0..=1.0).contains(&args.mosaic_probability)
        || !(0.0..=1.0).contains(&args.mixup_probability)
    {
        return Err(
            "size must be a positive multiple of 32; learning rate must be finite; steps, batch-size, prefetch, augmentation-workers, shuffle-buffer and log-every must be non-zero; mosaic-probability and mixup-probability must be in [0, 1]"
                .into(),
        );
    }
    let source = TrainingSource::open(args.source)?;
    let no_augmentation_steps = args.no_augmentation_steps.unwrap_or(args.steps / 20);
    let samples = ShufflePool::new(SourceSamples(source), args.shuffle_buffer, args.seed)?;
    let batches = PlannedBatches {
        samples,
        remaining: args.steps,
        step: 0,
        mosaic_steps: args.steps.saturating_sub(no_augmentation_steps),
        mosaic_probability: args.mosaic_probability,
        mixup_probability: args.mixup_probability,
        batch_size: args.batch_size,
        size: u32::try_from(args.size)?,
        rng: DataRng::new(args.seed),
    };
    // Host input preparation starts before XLA compilation so initial pool fill
    // and the first augmented batches are normally hidden by compilation.
    let pipeline = BoundedPipeline::from_iter(args.prefetch, batches)?.map_parallel_ordered(
        args.prefetch,
        args.augmentation_workers,
        |batch| prepare_batch(batch.groups, batch.size),
    )?;
    let batch = i64::try_from(args.batch_size)?;
    let client = unsafe { Client::load(&args.plugin) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compute_dtype = match args.compute_dtype {
        ComputeDtype::F32 => DType::F32,
        ComputeDtype::Bf16 => DType::BF16,
    };
    let training_config = YoloConfig::yolox_tiny(80).compute_dtype(compute_dtype);
    let training = Model::new(
        move |cx: Cx,
              image: Tensor,
              target_boxes: Tensor,
              target_classes: Tensor,
              target_valid: Tensor,
              learning_rate: Tensor,
              l1_weight: Tensor| {
            let image = image.cast(DType::F32)?;
            let outputs = yolo(cx, &image, &training_config)?;
            let assignments = yolox_simota(
                &outputs,
                &YoloTargets {
                    boxes: target_boxes,
                    classes: target_classes,
                    valid: target_valid,
                },
                YoloSimOtaConfig::default(),
            )?;
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
        ModelInput::new([batch, args.size, args.size, 3]).with_dtype(DType::U8),
        ModelInput::new([batch, MAX_TARGETS as i64, 4]),
        ModelInput::new([batch, MAX_TARGETS as i64, 80]),
        ModelInput::new([batch, MAX_TARGETS as i64, 1]),
        ModelInput::new([]),
        ModelInput::new([]),
    ))
    .training();
    let (trainable, mut training) = training.trace_resident_all()?;
    let checkpoint_schema = training.schema().clone();
    let metrics = Tensor::stack(&training.outputs()[..6], 0)?;
    let loss = training.outputs()[0].clone();
    let learning_rate = training.outputs()[6].clone();
    apply_model_sgd_with_learning_rate(
        &mut training,
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
    let train_compile_started = Instant::now();
    let program = training.compile_stateful_tensors(&mut compiler, &[metrics])?;
    let train_compile_ms = train_compile_started.elapsed().as_secs_f64() * 1e3;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let weights = checkpoint.load_model_schema(&client, &checkpoint_schema)?;
    let mut training_session = weights
        .initialize_raw(training.session(&program))?
        .initializer_seed(7)
        .build()?;
    let schedule = YoloLrSchedule::new(args.learning_rate, args.steps)
        .warmup_steps(args.warmup_steps.unwrap_or(args.steps / 60))
        .no_augmentation_steps(no_augmentation_steps);
    let mut completed = 0;
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
            load_ms,
        } = prepared?;
        let input = client.buffer(&[batch, args.size, args.size, 3], &pixels)?;
        let step_learning_rate = schedule.at(completed)?;
        let learning_rate = client.buffer(&[], &[step_learning_rate])?;
        let l1_weight_value = if no_augmentation_steps != 0
            && completed >= args.steps.saturating_sub(no_augmentation_steps)
        {
            1.0
        } else {
            0.0
        };
        let l1_weight = client.buffer(&[], &[l1_weight_value])?;
        let (target_boxes, target_classes, target_valid) =
            padded_targets(args.batch_size, MAX_TARGETS, 80, &targets)?;
        let target_boxes = client.buffer(&[batch, MAX_TARGETS as i64, 4], &target_boxes)?;
        let target_classes = client.buffer(&[batch, MAX_TARGETS as i64, 80], &target_classes)?;
        let target_valid = client.buffer(&[batch, MAX_TARGETS as i64, 1], &target_valid)?;
        let train_started = Instant::now();
        let outputs = training_session.run(&[
            &input,
            &target_boxes,
            &target_classes,
            &target_valid,
            &learning_rate,
            &l1_weight,
        ])?;
        let train_ms = train_started.elapsed().as_secs_f64() * 1e3;
        let losses = outputs[0].to_vec::<f32>()?;
        if losses.len() != 6 || losses.iter().any(|value| !value.is_finite()) {
            return Err("training produced invalid losses".into());
        }
        let foreground = losses[5] as usize;
        if foreground == 0 {
            return Err(format!(
                "SimOTA produced no foreground anchors for image {}",
                sample_ids.join(",")
            )
            .into());
        }
        completed += 1;
        if completed == 1 && args.cuda_profile {
            profiler = Some(CudaProfiler::start(&args.cuda_runtime)?);
        }
        if completed == 1 || completed % args.log_every == 0 || completed == args.steps {
            println!(
                "step={completed} image_id={} targets={} foreground={} total={:.6} box={:.6} objectness={:.6} classes={:.6} l1={:.6} lr={step_learning_rate:.8} wait_ms={wait_ms:.1} load_ms={load_ms:.1} train_ms={train_ms:.1}",
                sample_ids.join(","),
                targets.len(),
                foreground,
                losses[0],
                losses[1],
                losses[2],
                losses[3],
                losses[4],
            );
        }
    }
    drop(profiler);
    pipeline.finish()?;
    if completed != args.steps {
        return Err(format!(
            "only found {completed} annotated images for {} steps",
            args.steps
        )
        .into());
    }
    let run_seconds = run_started.elapsed().as_secs_f64();
    if let Some(output) = &args.output {
        let metadata = BTreeMap::from([
            ("format".to_owned(), "rxla-yolox-v1".to_owned()),
            ("steps".to_owned(), completed.to_string()),
        ])
        .into_iter()
        .collect();
        save_model_new(output, &training, &training_session, Some(&metadata))?;
    }
    println!(
        "completed={completed} images={} run_ms={:.1} images_per_second={:.1} train_compile_ms={train_compile_ms:.1}",
        completed * args.batch_size,
        run_seconds * 1e3,
        completed as f64 * args.batch_size as f64 / run_seconds,
    );
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
        self.remaining -= 1;
        self.step += 1;
        Some(Ok(AugmentationBatch {
            groups,
            size: self.size,
        }))
    }
}

fn prepare_batch(
    groups: Vec<AugmentationGroup>,
    size: u32,
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
                prepare_mosaic(samples, mixup, size, batch_index, rng)
            } else {
                let sample = samples
                    .into_iter()
                    .next()
                    .ok_or_else(|| "empty augmentation sample group".to_owned())?;
                prepare_single(sample, size, batch_index, rng)
            }
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    let mut pixels = Vec::with_capacity(prepared.len() * (size * size * 3) as usize);
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
    let mut image = decode_sample(&sample)?;
    let mut boxes = sample.boxes.clone();
    train_transform(&mut image, &mut boxes, rng);
    let (pixels, ratio) = letterbox(&image, size);
    let targets = detection_targets(&boxes, &sample.classes, ratio, batch);
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
            let resized = image::imageops::resize(&image, width, height, FilterType::Triangle);
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
        let crop = image::imageops::crop_imm(
            &resized,
            sx1 as u32,
            sy1 as u32,
            (lx2 - lx1) as u32,
            (ly2 - ly1) as u32,
        )
        .to_image();
        image::imageops::replace(&mut canvas, &crop, i64::from(lx1), i64::from(ly1));
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
    let mut transformed = boxes
        .into_iter()
        .map(|bbox| transform_box(bbox, matrix, size as f32))
        .collect::<Vec<_>>();
    if let Some(mixup) = mixup {
        ids.push(format!("mixup:{}", mixup.id));
        apply_mixup(&mut image, &mut transformed, &mut classes, mixup, size, rng)?;
    }
    train_transform(&mut image, &mut transformed, rng);
    let mut targets = detection_targets(&transformed, &classes, 1.0, batch);
    targets.retain(|target| target.cxcywh[2] > 1.0 && target.cxcywh[3] > 1.0);
    Ok((
        ids.join("+"),
        image
            .pixels()
            .flat_map(|pixel| pixel.0.into_iter().rev())
            .collect(),
        targets,
    ))
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
    let resized =
        image::imageops::resize(&source, resized_width, resized_height, FilterType::Triangle);
    let mut copy = RgbImage::from_pixel(size, size, Rgb([114; 3]));
    for (x, y, pixel) in resized.enumerate_pixels() {
        copy.put_pixel(x, y, *pixel);
    }

    let jitter = 0.5 + rng.uniform(5, 0);
    let width = (size as f32 * jitter) as u32;
    let height = (size as f32 * jitter) as u32;
    let mut copy = image::imageops::resize(&copy, width, height, FilterType::Triangle);
    let flipped = rng.uniform(5, 1) > 0.5;
    if flipped {
        image::imageops::flip_horizontal_in_place(&mut copy);
    }
    let x_range = width.saturating_sub(size);
    let y_range = height.saturating_sub(size);
    let x_offset = (rng.uniform(5, 2) * x_range as f32) as u32;
    let y_offset = (rng.uniform(5, 3) * y_range as f32) as u32;

    for y in 0..size {
        for x in 0..size {
            let copy_x = x + x_offset;
            let copy_y = y + y_offset;
            let overlay = if copy_x < width && copy_y < height {
                copy.get_pixel(copy_x, copy_y).0
            } else {
                [0; 3]
            };
            let origin = image.get_pixel(x, y).0;
            image.put_pixel(
                x,
                y,
                Rgb(std::array::from_fn(|channel| {
                    ((u16::from(origin[channel]) + u16::from(overlay[channel])) / 2) as u8
                })),
            );
        }
    }

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
    let image = image::load_from_memory(&sample.encoded_image)
        .map_err(|error| error.to_string())?
        .into_rgb8();
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
    for pixel in image.pixels_mut() {
        let [mut hue, mut saturation, mut value] = rgb_to_opencv_hsv(pixel.0);
        hue = (hue + hue_delta).rem_euclid(180.0);
        saturation = (saturation + saturation_delta).clamp(0.0, 255.0);
        value = (value + value_delta).clamp(0.0, 255.0);
        *pixel = Rgb(opencv_hsv_to_rgb([hue, saturation, value]));
    }
}

fn rgb_to_opencv_hsv([red, green, blue]: [u8; 3]) -> [f32; 3] {
    let red = f32::from(red) / 255.0;
    let green = f32::from(green) / 255.0;
    let blue = f32::from(blue) / 255.0;
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let hue = if delta == 0.0 {
        0.0
    } else if maximum == red {
        60.0 * ((green - blue) / delta).rem_euclid(6.0)
    } else if maximum == green {
        60.0 * ((blue - red) / delta + 2.0)
    } else {
        60.0 * ((red - green) / delta + 4.0)
    };
    [
        hue * 0.5,
        if maximum == 0.0 {
            0.0
        } else {
            delta / maximum * 255.0
        },
        maximum * 255.0,
    ]
}

fn opencv_hsv_to_rgb([hue, saturation, value]: [f32; 3]) -> [u8; 3] {
    let hue = hue * 2.0;
    let saturation = saturation / 255.0;
    let value = value / 255.0;
    let chroma = value * saturation;
    let section = hue / 60.0;
    let x = chroma * (1.0 - (section.rem_euclid(2.0) - 1.0).abs());
    let (red, green, blue) = match section as usize {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let offset = value - chroma;
    [red, green, blue].map(|channel| ((channel + offset) * 255.0).round() as u8)
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

fn warp_affine(image: &RgbImage, size: u32, matrix: [[f32; 3]; 2]) -> RgbImage {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let inverse = [
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ];
    let mut output = RgbImage::from_pixel(size, size, Rgb([114; 3]));
    for y in 0..size {
        for x in 0..size {
            let translated_x = x as f32 - matrix[0][2];
            let translated_y = y as f32 - matrix[1][2];
            let source_x = inverse[0][0] * translated_x + inverse[0][1] * translated_y;
            let source_y = inverse[1][0] * translated_x + inverse[1][1] * translated_y;
            if let Some(pixel) = image::imageops::interpolate_bilinear(image, source_x, source_y) {
                output.put_pixel(x, y, pixel);
            }
        }
    }
    output
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
                    || manifest.schema_sha256 != DETECTION_SCHEMA_SHA256
                {
                    return Err(format!(
                        "incompatible TFRecord/Fory manifest {}",
                        manifest_path.display()
                    )
                    .into());
                }
                Ok(Self::Tfrecord {
                    reader: TfrecordReader::open(input)?,
                    categories: category_map(COCO_CATEGORY_IDS),
                })
            }
        }
    }

    fn next_sample(&mut self) -> Result<Option<DetectionSample>> {
        match self {
            Self::Coco {
                document,
                images,
                position,
                annotations,
                categories,
            } => {
                let Some(image) = document.images.get(*position) else {
                    return Ok(None);
                };
                *position += 1;
                let mut boxes = Vec::new();
                let mut classes = Vec::new();
                for &index in annotations.get(&image.id).into_iter().flatten() {
                    let annotation = &document.annotations[index];
                    let [x, y, width, height] = annotation.bbox;
                    boxes.push([x, y, x + width, y + height]);
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
            Self::Tfrecord { reader, categories } => {
                let Some(payload) = reader.read_record()? else {
                    return Ok(None);
                };
                let row = from_row::<DetectionRecord>(payload)?;
                let boxes = row.boxes()?;
                let labels = row.labels()?;
                if boxes.len() != labels.len() * 4 {
                    return Err("Fory detection row has inconsistent boxes and labels".into());
                }
                let dimensions = [u32::try_from(row.width()?)?, u32::try_from(row.height()?)?];
                let boxes = (0..labels.len())
                    .map(|index| {
                        Ok([
                            boxes.get(index * 4)?,
                            boxes.get(index * 4 + 1)?,
                            boxes.get(index * 4 + 2)?,
                            boxes.get(index * 4 + 3)?,
                        ])
                    })
                    .collect::<std::result::Result<Vec<_>, fory::Error>>()?;
                let classes = labels
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
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut boxes = vec![0.0; batch * max_targets * 4];
    let mut one_hot = vec![0.0; batch * max_targets * classes];
    let mut valid = vec![0.0; batch * max_targets];
    let mut counts = vec![0; batch];
    for target in targets {
        if target.batch >= batch || target.class >= classes {
            return Err("target batch or class is out of range".into());
        }
        let slot = counts[target.batch];
        if slot == max_targets {
            continue;
        }
        counts[target.batch] += 1;
        let row = target.batch * max_targets + slot;
        boxes[row * 4..row * 4 + 4].copy_from_slice(&target.cxcywh);
        one_hot[row * classes + target.class] = 1.0;
        valid[row] = 1.0;
    }
    Ok((boxes, one_hot, valid))
}

fn letterbox(image: &RgbImage, size: u32) -> (Vec<u8>, f32) {
    let ratio = (size as f32 / image.width() as f32).min(size as f32 / image.height() as f32);
    let width = (image.width() as f32 * ratio) as u32;
    let height = (image.height() as f32 * ratio) as u32;
    let resized = image::imageops::resize(image, width, height, FilterType::Triangle);
    let mut output = vec![114; (size * size * 3) as usize];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = ((y * size + x) * 3) as usize;
        output[offset..offset + 3].copy_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    (output, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_are_padded_one_hot_and_capped_per_image() {
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
        let (boxes, classes, valid) = padded_targets(2, 1, 3, &targets).unwrap();
        assert_eq!(&boxes[4..8], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&classes[3..6], &[0.0, 0.0, 1.0]);
        assert_eq!(valid, [0.0, 1.0]);
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
    fn hsv_conversion_round_trips_primary_colors() {
        for color in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]] {
            assert_eq!(opencv_hsv_to_rgb(rgb_to_opencv_hsv(color)), color);
        }
    }
}
