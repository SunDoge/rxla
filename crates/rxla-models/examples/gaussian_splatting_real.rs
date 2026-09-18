use clap::Parser;
use image::{Rgb, RgbImage};
use rxla_core::{CacheLimits, Client, Compiler, Tracer};
use rxla_models::gaussian_splatting::{GaussianCamera, render_gaussians};
use serde::Deserialize;
use std::{cmp::Ordering, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    splat: PathBuf,
    #[arg(long)]
    cameras: PathBuf,
    #[arg(long, default_value_t = 0)]
    camera: usize,
    #[arg(long, default_value_t = 160)]
    width: u32,
    #[arg(long, default_value_t = 512)]
    gaussians: usize,
    #[arg(long, default_value_t = 20)]
    iterations: usize,
    #[arg(long, default_value = "gaussian-splatting-real.png")]
    output: PathBuf,
    #[arg(long)]
    plugin: Option<PathBuf>,
}

#[derive(Deserialize)]
struct CameraFile {
    img_name: String,
    width: u32,
    height: u32,
    position: [f32; 3],
    rotation: [[f32; 3]; 3],
    fx: f32,
    fy: f32,
}

struct Splat {
    mean: [f32; 3],
    scale: [f32; 3],
    color: [f32; 3],
    opacity: f32,
    quaternion: [f32; 4],
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("four-byte field"),
    )
}

fn load_splats(path: &PathBuf) -> Result<Vec<Splat>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 32 != 0 {
        return Err(format!("{} is not a 32-byte-record .splat file", path.display()).into());
    }
    let (records, remainder) = bytes.as_chunks::<32>();
    debug_assert!(remainder.is_empty());
    Ok(records
        .iter()
        .map(|row| {
            let mut quaternion = [0.; 4];
            for (output, input) in quaternion.iter_mut().zip(&row[28..32]) {
                *output = (*input as f32 - 128.) / 128.;
            }
            let norm = quaternion
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if norm > 0. {
                for value in &mut quaternion {
                    *value /= norm;
                }
            }
            Splat {
                mean: [f32_at(row, 0), f32_at(row, 4), f32_at(row, 8)],
                scale: [f32_at(row, 12), f32_at(row, 16), f32_at(row, 20)],
                color: [
                    row[24] as f32 / 255.,
                    row[25] as f32 / 255.,
                    row[26] as f32 / 255.,
                ],
                opacity: row[27] as f32 / 255.,
                quaternion,
            }
        })
        .collect())
}

fn mat_vec(matrix: &[[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
}

fn transpose(matrix: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn mat_mul(lhs: [[f32; 3]; 3], rhs: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| (0..3).map(|k| lhs[row][k] * rhs[k][column]).sum())
    })
}

fn quaternion_matrix([w, x, y, z]: [f32; 4]) -> [[f32; 3]; 3] {
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ]
}

fn world_covariance(splat: &Splat) -> [[f32; 3]; 3] {
    let rotation = quaternion_matrix(splat.quaternion);
    let scaled = std::array::from_fn(|row| {
        std::array::from_fn(|column| rotation[row][column] * splat.scale[column].powi(2))
    });
    mat_mul(scaled, transpose(rotation))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.width == 0 || args.gaussians == 0 || args.iterations == 0 {
        return Err("width, gaussians and iterations must be positive".into());
    }
    let cameras: Vec<CameraFile> = serde_json::from_slice(&fs::read(&args.cameras)?)?;
    let camera = cameras
        .get(args.camera)
        .ok_or("camera index is out of range")?;
    let height =
        ((args.width as f64 * camera.height as f64 / camera.width as f64).round() as u32).max(1);
    let scale_x = args.width as f32 / camera.width as f32;
    let scale_y = height as f32 / camera.height as f32;
    let focal_x = camera.fx * scale_x;
    let focal_y = camera.fy * scale_y;
    let center_x = args.width as f32 * 0.5;
    let center_y = height as f32 * 0.5;
    let world_to_camera = transpose(camera.rotation);

    let splats = load_splats(&args.splat)?;
    let total = splats.len();
    let mut visible = splats
        .into_iter()
        .filter_map(|splat| {
            let relative = std::array::from_fn(|axis| splat.mean[axis] - camera.position[axis]);
            let mean = mat_vec(&world_to_camera, relative);
            if mean[2] <= 0.01 {
                return None;
            }
            let x = focal_x * mean[0] / mean[2] + center_x;
            let y = focal_y * mean[1] / mean[2] + center_y;
            let margin = args.width.max(height) as f32 * 0.25;
            if x < -margin
                || x > args.width as f32 + margin
                || y < -margin
                || y > height as f32 + margin
            {
                return None;
            }
            let projected_scale = focal_x * splat.scale.into_iter().fold(0., f32::max) / mean[2];
            // Very large splats otherwise monopolize a small dense subset.
            // Cap their screen-size advantage so opaque surface detail survives.
            let score = splat.opacity * projected_scale.clamp(0.01, 8.0);
            let covariance = mat_mul(
                mat_mul(world_to_camera, world_covariance(&splat)),
                transpose(world_to_camera),
            );
            Some((score, mean, covariance, splat.color, splat.opacity))
        })
        .collect::<Vec<_>>();
    visible.sort_unstable_by(|lhs, rhs| rhs.0.partial_cmp(&lhs.0).unwrap_or(Ordering::Equal));
    visible.truncate(args.gaussians);
    if visible.is_empty() {
        return Err("selected camera has no visible Gaussians".into());
    }

    let mut means = Vec::with_capacity(visible.len() * 3);
    let mut covariances = Vec::with_capacity(visible.len() * 9);
    let mut colors = Vec::with_capacity(visible.len() * 3);
    let mut opacities = Vec::with_capacity(visible.len());
    for (_, mean, covariance, color, opacity) in &visible {
        means.extend_from_slice(mean);
        for row in covariance {
            covariances.extend_from_slice(row);
        }
        colors.extend_from_slice(color);
        opacities.push(*opacity);
    }
    let mut pixels = Vec::with_capacity(args.width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..args.width {
            pixels.extend_from_slice(&[x as f32 + 0.5, y as f32 + 0.5]);
        }
    }

    let plugin = args
        .plugin
        .or_else(|| std::env::var_os("PJRT_PLUGIN_PATH").map(PathBuf::from))
        .ok_or("pass --plugin or set PJRT_PLUGIN_PATH")?;
    let client = unsafe { Client::load(plugin) }?;
    let tracer = Tracer::default();
    let count = i64::try_from(visible.len())?;
    let means = tracer.constant(&[count, 3], &means)?;
    let covariances = tracer.constant(&[count, 3, 3], &covariances)?;
    let colors = tracer.constant(&[count, 3], &colors)?;
    let opacities = tracer.constant(&[count], &opacities)?;
    let pixels = tracer.constant(&[i64::from(height), i64::from(args.width), 2], &pixels)?;
    let background = tracer.constant(&[3], &[0., 0., 0.])?;
    let render = render_gaussians(
        &means,
        &covariances,
        &colors,
        &opacities,
        &pixels,
        &background,
        GaussianCamera::new(i64::from(args.width), i64::from(height), focal_x, focal_y),
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
    let mut image = RgbImage::new(args.width, height);
    for (pixel, color) in image.pixels_mut().zip(rgb) {
        *pixel = Rgb(color.map(|value| (value.clamp(0., 1.) * 255.).round() as u8));
    }
    image.save(&args.output)?;
    println!(
        "real 3DGS {} camera {} ({}): {}x{}, {}/{} visible-selected Gaussians, compile {:.3}s, mean {:.3}ms, output {}",
        args.splat.display(),
        args.camera,
        camera.img_name,
        args.width,
        height,
        visible.len(),
        total,
        compile_time.as_secs_f64(),
        elapsed.as_secs_f64() * 1e3 / args.iterations as f64,
        args.output.display()
    );
    Ok(())
}
