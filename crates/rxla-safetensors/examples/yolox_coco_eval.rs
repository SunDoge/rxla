//! Export YOLOX detections in the standard COCO result format.

use clap::Parser;
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image as FastImage, ImageRef},
};
use image::RgbImage;
use rxla_core::{Buffer, CacheLimits, Client, ClientOptions, Compiler};
use rxla_models::{YoloConfig, YoloDecodeConfig, yolo};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SIZE: u32 = 416;
const COCO_CATEGORY_IDS: [u64; 80] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
];

#[derive(Parser)]
#[command(about = "Export RXLA YOLOX-tiny predictions for COCO evaluation")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Inference checkpoint produced by conversion or training.
    weights: PathBuf,
    /// COCO instances_val JSON file.
    annotations: PathBuf,
    /// Directory containing validation images.
    images: PathBuf,
    /// JSON result consumed by pycocotools COCOeval.
    output: PathBuf,
    /// Evaluate only the first N images (useful for smoke tests).
    #[arg(long)]
    max_images: Option<usize>,
    /// Evaluate the packed EMA stored in a resumable training snapshot.
    #[arg(long)]
    ema: bool,
    /// Export this exact StableHLO program and its first-image ABI arguments.
    #[arg(long)]
    export_jax: Option<PathBuf>,
    #[arg(long, default_value_t = 0.01)]
    confidence: f32,
    #[arg(long, default_value_t = 0.65)]
    iou_threshold: f32,
    #[arg(long, default_value_t = 100)]
    log_every: usize,
    /// PJRT allocator reservation fraction; omit to use the plugin default.
    #[arg(long)]
    memory_fraction: Option<f32>,
    /// Whether PJRT preallocates its allocator reservation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    preallocate: bool,
}

#[derive(Deserialize)]
struct Coco {
    images: Vec<CocoImage>,
}

#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
}

#[derive(Serialize)]
struct CocoDetection {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
    score: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.log_every == 0
        || args
            .memory_fraction
            .is_some_and(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
    {
        return Err("log interval and memory fraction must be valid".into());
    }
    let document: Coco = serde_json::from_reader(BufReader::new(File::open(&args.annotations)?))?;
    let image_count = args
        .max_images
        .unwrap_or(document.images.len())
        .min(document.images.len());

    let mut client_options = ClientOptions::new().set("preallocate", args.preallocate);
    if let Some(memory_fraction) = args.memory_fraction {
        client_options = client_options.set("memory_fraction", memory_fraction);
    }
    let client = unsafe { Client::load_with_options(&args.plugin, &client_options) }?;
    let definition = Model::new(|cx: Cx, image| yolo(cx, &image, &YoloConfig::yolox_tiny(80)))
        .inputs(ModelInput::new([1, i64::from(SIZE), i64::from(SIZE), 3]))
        .inference();
    let (_, model) = definition.trace_resident_all()?;
    let lowered = args
        .export_jax
        .as_ref()
        .map(|_| model.prepare_stateful())
        .transpose()?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compiled = model.compile_stateful(&mut compiler)?;
    let mut checkpoint = SafeTensors::open(&args.weights)?;
    let format = checkpoint
        .metadata()
        .and_then(|metadata| metadata.get("format"))
        .map(String::as_str);
    if !matches!(
        format,
        Some("rxla-yolox-v1" | "rxla-yolox-training-v2" | "rxla-yolox-training-v3")
    ) {
        return Err(format!("unsupported YOLOX checkpoint format {format:?}").into());
    }
    let weights = if args.ema {
        if !matches!(
            format,
            Some("rxla-yolox-training-v2" | "rxla-yolox-training-v3")
        ) {
            return Err("--ema requires an RXLA YOLOX training snapshot".into());
        }
        checkpoint.load_model_ema_schema(&client, model.schema())?
    } else {
        checkpoint.load_model_schema(&client, model.schema())?
    };
    let mut session = weights.initialize(compiled.session())?.build()?;
    let decoder = YoloDecodeConfig::default()
        .confidence(args.confidence)
        .iou_threshold(args.iou_threshold);

    let started = Instant::now();
    let mut results = Vec::new();
    for (index, record) in document.images.iter().take(image_count).enumerate() {
        let image = image::open(args.images.join(&record.file_name))?.into_rgb8();
        let (input, ratio) = letterbox_bgr(&image)?;
        let input = client.buffer(&[1, 416, 416, 3], &input)?;
        if index == 0
            && let (Some(directory), Some(lowered)) = (&args.export_jax, &lowered)
        {
            let arguments = session.raw().execution_arguments(&[&input])?;
            export_jax_fixture(directory, lowered.lowered_program().code(), &arguments)?;
        }
        let outputs: [Buffer; 3] = session.run(&input)?;
        let heads = outputs
            .iter()
            .map(Buffer::to_vec::<f32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for detection in decoder.decode([
            (&heads[0], [1, 52, 52, 85], 8),
            (&heads[1], [1, 26, 26, 85], 16),
            (&heads[2], [1, 13, 13, 85], 32),
        ])? {
            let [x0, y0, x1, y1] = detection.xyxy.map(|coordinate| coordinate / ratio);
            let class = usize::try_from(detection.class)?;
            results.push(CocoDetection {
                image_id: record.id,
                category_id: COCO_CATEGORY_IDS[class],
                bbox: [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)],
                score: detection.score,
            });
        }
        if (index + 1) % args.log_every == 0 || index + 1 == image_count {
            eprintln!(
                "images={}/{} detections={} elapsed_s={:.1}",
                index + 1,
                image_count,
                results.len(),
                started.elapsed().as_secs_f64(),
            );
        }
    }
    serde_json::to_writer(BufWriter::new(File::create(&args.output)?), &results)?;
    println!(
        "images={image_count} detections={} elapsed_s={:.1} output={}",
        results.len(),
        started.elapsed().as_secs_f64(),
        args.output.display(),
    );
    Ok(())
}

fn export_jax_fixture(directory: &Path, stablehlo: &[u8], arguments: &[&Buffer]) -> Result<()> {
    fs::create_dir_all(directory)?;
    fs::write(directory.join("module.mlir"), stablehlo)?;
    let mut manifest = BufWriter::new(File::create(directory.join("arguments.txt"))?);
    for (index, argument) in arguments.iter().enumerate() {
        let dtype = argument.dtype()?;
        if dtype != rxla_core::DType::F32 {
            return Err(
                format!("JAX fixture argument {index} has unsupported dtype {dtype:?}").into(),
            );
        }
        let shape = argument.dimensions()?;
        writeln!(
            manifest,
            "{}",
            shape
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )?;
        let values = argument.to_vec::<f32>()?;
        let mut output =
            BufWriter::new(File::create(directory.join(format!("arg-{index:04}.f32")))?);
        for value in values {
            output.write_all(&value.to_le_bytes())?;
        }
        output.flush()?;
    }
    manifest.flush()?;
    fs::write(directory.join("dynamic-input.txt"), "0\n")?;
    fs::write(directory.join("visible-outputs.txt"), "3\n")?;
    Ok(())
}

fn letterbox_bgr(image: &RgbImage) -> Result<(Vec<f32>, f32)> {
    let ratio = (SIZE as f32 / image.width() as f32).min(SIZE as f32 / image.height() as f32);
    let width = (image.width() as f32 * ratio) as u32;
    let height = (image.height() as f32 * ratio) as u32;
    let source = ImageRef::new(
        image.width(),
        image.height(),
        image.as_raw(),
        PixelType::U8x3,
    )?;
    let mut resized = FastImage::new(width, height, PixelType::U8x3);
    Resizer::new().resize(
        &source,
        &mut resized,
        &ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear)),
    )?;
    let resized = resized.into_vec();
    let mut output = vec![114.0_f32; (SIZE * SIZE * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let source = (y * width as usize + x) * 3;
            let target = (y * SIZE as usize + x) * 3;
            output[target..target + 3].copy_from_slice(&[
                f32::from(resized[source + 2]),
                f32::from(resized[source + 1]),
                f32::from(resized[source]),
            ]);
        }
    }
    Ok((output, ratio))
}
