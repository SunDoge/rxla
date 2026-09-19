//! Train YOLOX on real COCO samples with host SimOTA and resident momentum SGD.

use clap::{Parser, Subcommand};
use fory::{ForyRow, from_row};
use image::{RgbImage, imageops::FilterType};
use rayon::prelude::*;
use rxla_core::{Buffer, CacheLimits, Client, Compiler, Tensor};
use rxla_models::{YoloAssignments, YoloConfig, YoloLossConfig, yolo, yolox_loss};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::{SafeTensors, save_model_new};
use rxla_tfrecord::Reader as TfrecordReader;
use rxla_train::{
    BoundedPipeline, DecodedYoloBatch, DetectionTarget, SgdOptions, ShufflePool, SimOtaConfig,
    YoloHead, YoloLrSchedule, apply_model_sgd_with_learning_rate,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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
    #[arg(long, default_value_t = 2)]
    prefetch: usize,
    /// Number of sequential samples retained for deterministic pool shuffle.
    #[arg(long, default_value_t = 1024)]
    shuffle_buffer: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 0.00015625)]
    learning_rate: f32,
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
    pixels: Vec<f32>,
    targets: Vec<DetectionTarget>,
    load_ms: f64,
}

struct SourceSamples(TrainingSource);

struct PreparedBatches<I> {
    samples: I,
    remaining: usize,
    batch_size: usize,
    size: u32,
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
        || args.shuffle_buffer == 0
    {
        return Err(
            "size must be a positive multiple of 32; learning rate must be finite; steps, batch-size, prefetch, shuffle-buffer and log-every must be non-zero"
                .into(),
        );
    }
    let source = TrainingSource::open(args.source)?;
    let batch = i64::try_from(args.batch_size)?;
    let client = unsafe { Client::load(&args.plugin) }?;
    let config = YoloConfig::yolox_tiny(80);
    let inference = Model::new(move |cx: Cx, image| yolo(cx, &image, &config))
        .inputs(ModelInput::new([batch, args.size, args.size, 3]))
        .inference();
    let (_, inference) = inference.trace_resident_all()?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let inference_program = inference.compile_stateful_program(&mut compiler)?;
    let anchors = head_shapes(1, args.size, 85)
        .iter()
        .map(|shape| shape[1] * shape[2])
        .sum::<usize>() as i64;
    let training_config = YoloConfig::yolox_tiny(80);
    let training = Model::new(
        move |cx: Cx,
              image: Tensor,
              boxes: Tensor,
              objectness: Tensor,
              classes: Tensor,
              learning_rate: Tensor| {
            let outputs = yolo(cx, &image, &training_config)?;
            let loss = yolox_loss(
                &outputs,
                &YoloAssignments {
                    boxes,
                    objectness,
                    classes,
                },
                YoloLossConfig::default(),
            )?;
            Ok::<_, rxla_models::Error>((
                loss.total,
                loss.boxes,
                loss.objectness,
                loss.classes,
                learning_rate,
            ))
        },
    )
    .inputs((
        ModelInput::new([batch, args.size, args.size, 3]),
        ModelInput::new([batch, anchors, 4]),
        ModelInput::new([batch, anchors, 1]),
        ModelInput::new([batch, anchors, 80]),
        ModelInput::new([]),
    ))
    .training();
    let (trainable, mut training) = training.trace_resident_all()?;
    let checkpoint_schema = training.schema().clone();
    let metrics = Tensor::stack(&training.outputs()[..4], 0)?;
    let loss = training.outputs()[0].clone();
    let learning_rate = training.outputs()[4].clone();
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
        .no_augmentation_steps(args.no_augmentation_steps.unwrap_or(args.steps / 20));
    let samples = ShufflePool::new(SourceSamples(source), args.shuffle_buffer, args.seed)?;
    let batches = PreparedBatches {
        samples,
        remaining: args.steps,
        batch_size: args.batch_size,
        size: u32::try_from(args.size)?,
    };
    let pipeline = BoundedPipeline::from_iter(args.prefetch, batches)?;
    let mut completed = 0;
    let run_started = Instant::now();
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

        // Buffer clones share their PJRT allocation. The inference detour sees
        // the current weights while the original snapshot retains SGD state.
        let snapshot = training.take_session(training_session)?;
        let mut inference_session = snapshot
            .clone()
            .restore_model(inference.session(&inference_program))?
            .build()?;
        let infer_started = Instant::now();
        let outputs: [Buffer; 3] =
            inference_session
                .run(&[&input])?
                .try_into()
                .map_err(|outputs: Vec<Buffer>| {
                    format!("inference returned {} outputs instead of 3", outputs.len())
                })?;
        let infer_ms = infer_started.elapsed().as_secs_f64() * 1e3;
        training_session = snapshot
            .restore_model(training.session(&program))?
            .build()?;

        let match_started = Instant::now();
        let heads = outputs
            .iter()
            .map(Buffer::to_vec::<f32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let shapes = head_shapes(batch, args.size, 85);
        let decoded = DecodedYoloBatch::from_heads([
            YoloHead::new(&heads[0], shapes[0]).stride(8),
            YoloHead::new(&heads[1], shapes[1]).stride(16),
            YoloHead::new(&heads[2], shapes[2]).stride(32),
        ])?;
        let assignments = SimOtaConfig::default().assign(&decoded.input(), &targets)?;
        let match_ms = match_started.elapsed().as_secs_f64() * 1e3;
        if assignments.foreground == 0 {
            return Err(format!(
                "SimOTA produced no foreground anchors for image {}",
                sample_ids.join(",")
            )
            .into());
        }
        let boxes = client.buffer(&[batch, anchors, 4], &assignments.boxes)?;
        let objectness = client.buffer(&[batch, anchors, 1], &assignments.objectness)?;
        let classes = client.buffer(&[batch, anchors, 80], &assignments.classes)?;
        let train_started = Instant::now();
        let outputs =
            training_session.run(&[&input, &boxes, &objectness, &classes, &learning_rate])?;
        let train_ms = train_started.elapsed().as_secs_f64() * 1e3;
        let losses = outputs[0].to_vec::<f32>()?;
        if losses.len() != 4 || losses.iter().any(|value| !value.is_finite()) {
            return Err("training produced invalid losses".into());
        }
        completed += 1;
        if completed == 1 || completed % args.log_every == 0 || completed == args.steps {
            println!(
                "step={completed} image_id={} targets={} foreground={} total={:.6} box={:.6} objectness={:.6} classes={:.6} lr={step_learning_rate:.8} wait_ms={wait_ms:.1} load_ms={load_ms:.1} infer_ms={infer_ms:.1} match_ms={match_ms:.1} train_ms={train_ms:.1}",
                sample_ids.join(","),
                targets.len(),
                assignments.foreground,
                losses[0],
                losses[1],
                losses[2],
                losses[3],
            );
        }
    }
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

impl<I> Iterator for PreparedBatches<I>
where
    I: Iterator<Item = std::result::Result<DetectionSample, String>>,
{
    type Item = std::result::Result<PreparedBatch, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let load_started = Instant::now();
        let mut samples = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.samples.next() {
                Some(Ok(sample)) => samples.push(sample),
                None => {
                    self.remaining = 0;
                    return Some(Err(format!(
                        "dataset ended with only {} of {} samples in the next batch",
                        samples.len(),
                        self.batch_size
                    )));
                }
                Some(Err(error)) => {
                    self.remaining = 0;
                    return Some(Err(error));
                }
            }
        }
        self.remaining -= 1;
        Some(prepare_batch(samples, self.size, load_started))
    }
}

fn prepare_batch(
    samples: Vec<DetectionSample>,
    size: u32,
    load_started: Instant,
) -> std::result::Result<PreparedBatch, String> {
    let prepared = samples
        .into_par_iter()
        .enumerate()
        .map(|(batch_index, sample)| {
            let image = image::load_from_memory(&sample.encoded_image)
                .map_err(|error| error.to_string())?
                .into_rgb8();
            if sample
                .dimensions
                .is_some_and(|dimensions| dimensions != [image.width(), image.height()])
            {
                return Err("image dimensions do not match Fory metadata".to_owned());
            }
            let (pixels, ratio) = letterbox(&image, size);
            let targets = sample.targets(ratio, batch_index);
            Ok((sample.id, pixels, targets))
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

impl DetectionSample {
    fn targets(&self, ratio: f32, batch: usize) -> Vec<DetectionTarget> {
        self.boxes
            .iter()
            .zip(&self.classes)
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

fn head_shapes(batch: i64, size: i64, channels: usize) -> [[usize; 4]; 3] {
    [8, 16, 32].map(|stride| {
        [
            batch as usize,
            (size / stride) as usize,
            (size / stride) as usize,
            channels,
        ]
    })
}

fn letterbox(image: &RgbImage, size: u32) -> (Vec<f32>, f32) {
    let ratio = (size as f32 / image.width() as f32).min(size as f32 / image.height() as f32);
    let width = (image.width() as f32 * ratio) as u32;
    let height = (image.height() as f32 * ratio) as u32;
    let resized = image::imageops::resize(image, width, height, FilterType::Triangle);
    let mut output = vec![114.0; (size * size * 3) as usize];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = ((y * size + x) * 3) as usize;
        output[offset..offset + 3].copy_from_slice(&pixel.0.map(f32::from));
    }
    (output, ratio)
}
