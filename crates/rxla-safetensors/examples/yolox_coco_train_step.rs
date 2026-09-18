//! Train YOLOX on real COCO samples with host SimOTA and a resident Adam session.

use clap::Parser;
use image::{RgbImage, imageops::FilterType};
use rxla_core::{Buffer, CacheLimits, Client, Compiler, Tensor};
use rxla_models::{YoloAssignments, YoloConfig, YoloLossConfig, yolo, yolox_loss};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::{SafeTensors, save_model_new};
use rxla_train::{
    AdamOptions, DecodedYoloBatch, DetectionTarget, SimOtaConfig, YoloHead, apply_model_adam,
};
use serde::Deserialize;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Run consecutive real COCO YOLOX training steps")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Checkpoint produced by benchmarks/convert_yolox.py.
    weights: PathBuf,
    /// COCO instances JSON.
    annotations: PathBuf,
    /// Image directory referenced by the instances JSON.
    images: PathBuf,
    /// COCO image ID at which to start iterating.
    image_id: u64,
    /// Number of consecutive annotated images to train on.
    #[arg(long, default_value_t = 1)]
    steps: usize,
    #[arg(long, default_value_t = 64)]
    size: i64,
    #[arg(long, default_value_t = 1e-5)]
    learning_rate: f32,
    /// Write inference-ready trained parameters and BatchNorm state here.
    #[arg(long)]
    output: Option<PathBuf>,
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.size <= 0 || args.size % 32 != 0 || !args.learning_rate.is_finite() || args.steps == 0 {
        return Err(
            "size must be a positive multiple of 32, learning rate finite, and steps non-zero"
                .into(),
        );
    }
    let coco: Coco = serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(
        &args.annotations,
    )?))?;
    let start = coco
        .images
        .iter()
        .position(|image| image.id == args.image_id)
        .ok_or("image ID is absent from annotations")?;
    let mut category_ids = coco
        .categories
        .iter()
        .map(|category| category.id)
        .collect::<Vec<_>>();
    category_ids.sort_unstable();
    let categories = category_ids
        .into_iter()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let client = unsafe { Client::load(&args.plugin) }?;
    let config = YoloConfig::yolox_tiny(80);
    let inference = Model::new(move |cx: Cx, image| yolo(cx, &image, &config))
        .inputs(ModelInput::new([1, args.size, args.size, 3]))
        .inference();
    let (_, inference) = inference.trace_resident_all()?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let inference_program = inference.compile_stateful_program(&mut compiler)?;
    let anchors = head_shapes(args.size, 85)
        .iter()
        .map(|shape| shape[1] * shape[2])
        .sum::<usize>() as i64;
    let training_config = YoloConfig::yolox_tiny(80);
    let training = Model::new(
        move |cx: Cx, image: Tensor, boxes: Tensor, objectness: Tensor, classes: Tensor| {
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
            Ok::<_, rxla_models::Error>((loss.total, loss.boxes, loss.objectness, loss.classes))
        },
    )
    .inputs((
        ModelInput::new([1, args.size, args.size, 3]),
        ModelInput::new([1, anchors, 4]),
        ModelInput::new([1, anchors, 1]),
        ModelInput::new([1, anchors, 80]),
    ))
    .training();
    let (trainable, mut training) = training.trace_resident_all()?;
    let checkpoint_schema = training.schema().clone();
    let metrics = Tensor::stack(training.outputs(), 0)?;
    let loss = training.outputs()[0].clone();
    apply_model_adam(
        &mut training,
        &trainable,
        &loss,
        AdamOptions::default().learning_rate(args.learning_rate),
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
    let mut completed = 0;
    for image in &coco.images[start..] {
        let original = image::open(args.images.join(&image.file_name))?.into_rgb8();
        let (pixels, ratio) = letterbox(&original, args.size as u32);
        let targets = targets_for_image(&coco, &categories, image.id, ratio)?;
        if targets.is_empty() {
            continue;
        }
        let input = client.buffer(&[1, args.size, args.size, 3], &pixels)?;

        // Buffer clones share their PJRT allocation. The inference detour sees
        // the current weights while the original snapshot retains Adam state.
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

        let heads = outputs
            .iter()
            .map(Buffer::to_vec::<f32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let shapes = head_shapes(args.size, 85);
        let decoded = DecodedYoloBatch::from_heads([
            YoloHead::new(&heads[0], shapes[0]).stride(8),
            YoloHead::new(&heads[1], shapes[1]).stride(16),
            YoloHead::new(&heads[2], shapes[2]).stride(32),
        ])?;
        let assignments = SimOtaConfig::default().assign(&decoded.input(), &targets)?;
        if assignments.foreground == 0 {
            return Err(format!(
                "SimOTA produced no foreground anchors for image {}",
                image.id
            )
            .into());
        }
        let boxes = client.buffer(&[1, anchors, 4], &assignments.boxes)?;
        let objectness = client.buffer(&[1, anchors, 1], &assignments.objectness)?;
        let classes = client.buffer(&[1, anchors, 80], &assignments.classes)?;
        let train_started = Instant::now();
        let outputs = training_session.run(&[&input, &boxes, &objectness, &classes])?;
        let train_ms = train_started.elapsed().as_secs_f64() * 1e3;
        let losses = outputs[0].to_vec::<f32>()?;
        if losses.len() != 4 || losses.iter().any(|value| !value.is_finite()) {
            return Err("training produced invalid losses".into());
        }
        completed += 1;
        println!(
            "step={completed} image_id={} targets={} foreground={} total={:.6} box={:.6} objectness={:.6} classes={:.6} infer_ms={infer_ms:.1} train_ms={train_ms:.1}",
            image.id,
            targets.len(),
            assignments.foreground,
            losses[0],
            losses[1],
            losses[2],
            losses[3],
        );
        if completed == args.steps {
            break;
        }
    }
    if completed != args.steps {
        return Err(format!(
            "only found {completed} annotated images for {} steps",
            args.steps
        )
        .into());
    }
    if let Some(output) = &args.output {
        let metadata = BTreeMap::from([
            ("format".to_owned(), "rxla-yolox-v1".to_owned()),
            ("steps".to_owned(), completed.to_string()),
        ])
        .into_iter()
        .collect();
        save_model_new(output, &training, &training_session, Some(&metadata))?;
    }
    println!("completed={completed} train_compile_ms={train_compile_ms:.1}");
    Ok(())
}

fn targets_for_image(
    coco: &Coco,
    categories: &BTreeMap<u64, usize>,
    image_id: u64,
    ratio: f32,
) -> Result<Vec<DetectionTarget>> {
    coco.annotations
        .iter()
        .filter(|annotation| annotation.image_id == image_id && annotation.iscrowd == 0)
        .map(|annotation| {
            let [x, y, width, height] = annotation.bbox;
            Ok(DetectionTarget {
                batch: 0,
                class: *categories
                    .get(&annotation.category_id)
                    .ok_or("annotation references an unknown category")?,
                cxcywh: [
                    (x + width * 0.5) * ratio,
                    (y + height * 0.5) * ratio,
                    width * ratio,
                    height * ratio,
                ],
            })
        })
        .collect()
}

fn head_shapes(size: i64, channels: usize) -> [[usize; 4]; 3] {
    [8, 16, 32].map(|stride| {
        [
            1,
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
