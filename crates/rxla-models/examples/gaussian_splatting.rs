use clap::Parser;
use image::{Rgb, RgbImage};
use rxla_core::{CacheLimits, Client, Compiler, Tracer};
use rxla_models::gaussian_splatting::{GaussianCamera, render_gaussians};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 256)]
    width: u32,
    #[arg(long, default_value_t = 256)]
    height: u32,
    #[arg(long, default_value_t = 48)]
    gaussians: usize,
    #[arg(long, default_value_t = 20)]
    iterations: usize,
    #[arg(long, default_value = "gaussian-splatting.png")]
    output: PathBuf,
    #[arg(long)]
    plugin: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.width == 0 || args.height == 0 || args.gaussians == 0 || args.iterations == 0 {
        return Err("width, height, gaussians and iterations must be positive".into());
    }
    let plugin = args
        .plugin
        .or_else(|| std::env::var_os("PJRT_PLUGIN_PATH").map(PathBuf::from))
        .ok_or("pass --plugin or set PJRT_PLUGIN_PATH")?;
    let client = unsafe { Client::load(plugin) }?;
    let tracer = Tracer::default();
    let count = i64::try_from(args.gaussians)?;

    let mut means = Vec::with_capacity(args.gaussians * 3);
    let mut covariances = Vec::with_capacity(args.gaussians * 9);
    let mut colors = Vec::with_capacity(args.gaussians * 3);
    let mut opacities = Vec::with_capacity(args.gaussians);
    for index in 0..args.gaussians {
        let t = index as f32 / args.gaussians as f32;
        let angle = t * std::f32::consts::TAU * 3.0;
        let radius = 0.15 + 0.7 * t;
        means.extend_from_slice(&[radius * angle.cos(), radius * angle.sin(), 2.0 + 1.5 * t]);
        let scale = 0.012 + 0.018 * (1.0 - t);
        covariances.extend_from_slice(&[
            scale,
            0.,
            0.,
            0.,
            scale * (0.6 + 0.8 * t),
            0.,
            0.,
            0.,
            scale,
        ]);
        colors.extend_from_slice(&[
            0.5 + 0.5 * angle.cos(),
            0.5 + 0.5 * (angle + 2.094).cos(),
            0.5 + 0.5 * (angle + 4.189).cos(),
        ]);
        opacities.push(0.72);
    }
    let mut pixels = Vec::with_capacity(args.width as usize * args.height as usize * 2);
    for y in 0..args.height {
        for x in 0..args.width {
            pixels.extend_from_slice(&[x as f32 + 0.5, y as f32 + 0.5]);
        }
    }
    let means = tracer.constant([count, 3], &means)?;
    let covariances = tracer.constant([count, 3, 3], &covariances)?;
    let colors = tracer.constant([count, 3], &colors)?;
    let opacities = tracer.constant([count], &opacities)?;
    let pixels = tracer.constant([i64::from(args.height), i64::from(args.width), 2], &pixels)?;
    let background = tracer.constant([3], [0.025, 0.035, 0.06])?;
    let focal = args.width.min(args.height) as f32 * 0.9;
    let render = render_gaussians(
        &means,
        &covariances,
        &colors,
        &opacities,
        &pixels,
        &background,
        GaussianCamera::new(i64::from(args.width), i64::from(args.height), focal, focal),
    )?;
    let mut compiler = Compiler::new(client, CacheLimits::default());
    let compile_start = Instant::now();
    let executable = compiler.compile_many(&tracer, &[render.color, render.alpha])?;
    let compile_time = compile_start.elapsed();
    executable.submit(&[])?.wait()?;
    let start = Instant::now();
    let mut outputs = Vec::new();
    for _ in 0..args.iterations {
        outputs = executable.submit(&[])?.wait()?;
    }
    let elapsed = start.elapsed();
    let rgb = outputs[0].to_vec::<f32>()?;
    let (rgb, remainder) = rgb.as_chunks::<3>();
    debug_assert!(remainder.is_empty());
    let mut image = RgbImage::new(args.width, args.height);
    for (pixel, color) in image.pixels_mut().zip(rgb) {
        *pixel = Rgb([
            (color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]);
    }
    image.save(&args.output)?;
    println!(
        "pure-XLA 3DGS: {}x{}, {} gaussians, compile {:.3}s, mean {:.3}ms, output {}",
        args.width,
        args.height,
        args.gaussians,
        compile_time.as_secs_f64(),
        elapsed.as_secs_f64() * 1e3 / args.iterations as f64,
        args.output.display()
    );
    Ok(())
}
