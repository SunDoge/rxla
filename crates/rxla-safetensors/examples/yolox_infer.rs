//! Run converted official YOLOX-tiny weights on one image.

use clap::Parser;
use image::{Rgb, RgbImage, imageops::FilterType};
use rxla_core::{Buffer, CacheLimits, Client, Compiler};
use rxla_models::{YoloConfig, YoloDecodeConfig, yolo};
use rxla_nn::{Cx, Model, ModelInput};
use rxla_safetensors::SafeTensors;
use std::{path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SIZE: u32 = 416;

#[derive(Parser)]
#[command(about = "Run RXLA YOLOX-tiny inference on an image")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Checkpoint produced by benchmarks/convert_yolox.py.
    weights: PathBuf,
    /// JPEG or PNG input image.
    image: PathBuf,
    /// Annotated PNG output image.
    output: PathBuf,
    #[arg(long, default_value_t = 0.25)]
    confidence: f32,
    #[arg(long, default_value_t = 0.45)]
    iou_threshold: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let client = unsafe { Client::load(&args.plugin) }?;
    let definition = Model::new(|cx: Cx, image| yolo(cx, &image, &YoloConfig::yolox_tiny(80)))
        .inputs(ModelInput::new([1, i64::from(SIZE), i64::from(SIZE), 3]))
        .inference();
    let (_, model) = definition.trace_resident_all()?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let compile_started = Instant::now();
    let compiled = model.compile_stateful(&mut compiler)?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1e3;

    let mut checkpoint = SafeTensors::open(&args.weights)?;
    checkpoint.require_metadata(&[("format", "rxla-yolox-v1")])?;
    let load_started = Instant::now();
    let weights = checkpoint.load_model_schema(&client, model.schema())?;
    let mut session = weights.initialize(compiled.session())?.build()?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;

    let mut original = image::open(&args.image)?.into_rgb8();
    let ratio = (SIZE as f32 / original.width() as f32).min(SIZE as f32 / original.height() as f32);
    let resized_width = (original.width() as f32 * ratio) as u32;
    let resized_height = (original.height() as f32 * ratio) as u32;
    let resized = image::imageops::resize(
        &original,
        resized_width,
        resized_height,
        FilterType::Triangle,
    );
    let mut input = vec![114.0_f32; (SIZE * SIZE * 3) as usize];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = ((y * SIZE + x) * 3) as usize;
        input[offset..offset + 3].copy_from_slice(&[
            f32::from(pixel[2]),
            f32::from(pixel[1]),
            f32::from(pixel[0]),
        ]);
    }
    let input = client.buffer(&[1, i64::from(SIZE), i64::from(SIZE), 3], &input)?;
    let inference_started = Instant::now();
    let outputs: [Buffer; 3] = session.run(&input)?;
    let inference_ms = inference_started.elapsed().as_secs_f64() * 1e3;
    let heads = outputs
        .iter()
        .map(Buffer::to_vec::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let detections = YoloDecodeConfig::default()
        .confidence(args.confidence)
        .iou_threshold(args.iou_threshold)
        .decode([
            (&heads[0], [1, 52, 52, 85], 8),
            (&heads[1], [1, 26, 26, 85], 16),
            (&heads[2], [1, 13, 13, 85], 32),
        ])?;

    for detection in &detections {
        let [x0, y0, x1, y1] = detection.xyxy.map(|coordinate| coordinate / ratio);
        draw_rectangle(&mut original, [x0, y0, x1, y1], color(detection.class));
        println!(
            "class={} score={:.4} xyxy=[{x0:.1}, {y0:.1}, {x1:.1}, {y1:.1}]",
            detection.class, detection.score
        );
    }
    original.save(&args.output)?;
    println!(
        "detections={} compile_ms={compile_ms:.1} load_ms={load_ms:.1} inference_ms={inference_ms:.3} output={}",
        detections.len(),
        args.output.display()
    );
    Ok(())
}

fn color(class: i32) -> Rgb<u8> {
    let value = class as u32;
    Rgb([
        (37 * value + 97) as u8,
        (17 * value + 191) as u8,
        (29 * value + 53) as u8,
    ])
}

fn draw_rectangle(image: &mut RgbImage, xyxy: [f32; 4], color: Rgb<u8>) {
    let clamp_x = |value: f32| value.clamp(0.0, image.width().saturating_sub(1) as f32) as u32;
    let clamp_y = |value: f32| value.clamp(0.0, image.height().saturating_sub(1) as f32) as u32;
    let [x0, y0, x1, y1] = [
        clamp_x(xyxy[0]),
        clamp_y(xyxy[1]),
        clamp_x(xyxy[2]),
        clamp_y(xyxy[3]),
    ];
    for thickness in 0..2 {
        let left = x0.saturating_add(thickness).min(x1);
        let right = x1.saturating_sub(thickness).max(x0);
        let top = y0.saturating_add(thickness).min(y1);
        let bottom = y1.saturating_sub(thickness).max(y0);
        for x in left..=right {
            image.put_pixel(x, top, color);
            image.put_pixel(x, bottom, color);
        }
        for y in top..=bottom {
            image.put_pixel(left, y, color);
            image.put_pixel(right, y, color);
        }
    }
}
