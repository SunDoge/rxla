//! Execute one real COCO YOLOX SimOTA + Adam training step.

use clap::Parser;
use image::{RgbImage, imageops::FilterType};
use rxla_core::{Buffer, CacheLimits, Client, Compiler, Tensor};
use rxla_models::{YoloAssignments, YoloConfig, YoloLossConfig, yolo, yolox_loss};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use rxla_train::{
    AdamOptions, DecodedYoloBatch, DetectionTarget, SimOtaConfig, YoloHead, apply_model_adam,
};
use serde::Deserialize;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Run one real COCO YOLOX training step")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Checkpoint produced by benchmarks/convert_yolox.py.
    weights: PathBuf,
    /// COCO instances JSON.
    annotations: PathBuf,
    /// Image directory referenced by the instances JSON.
    images: PathBuf,
    /// COCO image ID to train on.
    image_id: u64,
    #[arg(long, default_value_t = 64)]
    size: i64,
    #[arg(long, default_value_t = 1e-5)]
    learning_rate: f32,
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
    if args.size <= 0 || args.size % 32 != 0 || !args.learning_rate.is_finite() {
        return Err("size must be a positive multiple of 32 and learning rate finite".into());
    }
    let coco: Coco = serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(
        &args.annotations,
    )?))?;
    let image = coco
        .images
        .iter()
        .find(|image| image.id == args.image_id)
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
    let original = image::open(args.images.join(&image.file_name))?.into_rgb8();
    let (pixels, ratio) = letterbox(&original, args.size as u32);
    let targets = coco
        .annotations
        .iter()
        .filter(|annotation| annotation.image_id == args.image_id && annotation.iscrowd == 0)
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
        .collect::<Result<Vec<_>>>()?;
    if targets.is_empty() {
        return Err("selected image has no non-crowd annotations".into());
    }

    let client = unsafe { Client::load(&args.plugin) }?;
    let input = client.buffer(&[1, args.size, args.size, 3], &pixels)?;
    let config = YoloConfig::yolox_tiny(80);
    let inference = Model::new(move |cx: Cx, image| yolo(cx, &image, &config))
        .inputs(ModelInput::new([1, args.size, args.size, 3]))
        .inference();
    let (_, inference) = inference.trace_resident_all()?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compiled = inference.compile_stateful(&mut compiler)?;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let weights = checkpoint.load_model_schema(&client, inference.schema())?;
    let mut session = weights.initialize(compiled.session())?.build()?;
    let outputs: [Buffer; 3] = session.run(&input)?;
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
        return Err("SimOTA produced no foreground anchors".into());
    }

    let anchors = assignments.anchors as i64;
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
    let compile_started = Instant::now();
    let program = training.compile_stateful_tensors(&mut compiler, &[metrics])?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let weights = checkpoint.load_model_schema(&client, &checkpoint_schema)?;
    let mut training_session = weights
        .initialize_raw(training.session(&program))?
        .initializer_seed(7)
        .build()?;
    let boxes = client.buffer(&[1, anchors, 4], &assignments.boxes)?;
    let objectness = client.buffer(&[1, anchors, 1], &assignments.objectness)?;
    let classes = client.buffer(&[1, anchors, 80], &assignments.classes)?;
    let execute_started = Instant::now();
    let outputs = training_session.run(&[&input, &boxes, &objectness, &classes])?;
    let execute_ms = execute_started.elapsed().as_secs_f64() * 1e3;
    let losses = outputs[0].to_vec::<f32>()?;
    if losses.len() != 4 || losses.iter().any(|value| !value.is_finite()) {
        return Err("training produced invalid losses".into());
    }
    println!(
        "image_id={} targets={} foreground={} total={:.6} box={:.6} objectness={:.6} classes={:.6} compile_ms={compile_ms:.1} execute_ms={execute_ms:.1}",
        args.image_id,
        targets.len(),
        assignments.foreground,
        losses[0],
        losses[1],
        losses[2],
        losses[3],
    );
    Ok(())
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
