//! Tensor-first fused image augmentation benchmark.
//!
//! Compares an independent scalar host implementation with one reusable RXLA
//! TensorFunction. Resident timing excludes upload/compilation and downloads
//! only a scalar checksum; end-to-end timing includes host Tensor construction,
//! upload, execution, and scalar download.
use clap::Parser;
use rxla_core::{DType, Runtime, Tensor, TensorFunction};
use std::{hint::black_box, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const INPUT: usize = 320;
const RESIZED: usize = 286;
const OUTPUT: usize = 256;
const CHANNELS: usize = 3;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STDDEV: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Parser)]
struct Args {
    /// Trusted PJRT plugin shared library.
    #[arg(long)]
    plugin: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 50)]
    runs: usize,
    #[arg(long, default_value_t = 5)]
    warmups: usize,
}

fn input_data(batch: usize) -> Vec<u8> {
    (0..batch * INPUT * INPUT * CHANNELS)
        .map(|index| ((index * 37 + index / 101) % 256) as u8)
        .collect()
}

fn host_pipeline(input: &[u8], batch: usize) -> f32 {
    let mut resized = vec![0.0f32; batch * RESIZED * RESIZED * CHANNELS];
    let scale = INPUT as f64 / RESIZED as f64;
    for n in 0..batch {
        for oy in 0..RESIZED {
            let sy = ((oy as f64 + 0.5) * scale - 0.5).clamp(0.0, INPUT as f64 - 1.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(INPUT - 1);
            let wy = (sy - y0 as f64) as f32;
            for ox in 0..RESIZED {
                let sx = ((ox as f64 + 0.5) * scale - 0.5).clamp(0.0, INPUT as f64 - 1.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(INPUT - 1);
                let wx = (sx - x0 as f64) as f32;
                for channel in 0..CHANNELS {
                    let at = |y, x| {
                        input[((n * INPUT + y) * INPUT + x) * CHANNELS + channel] as f32 / 255.0
                    };
                    let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * wx;
                    let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * wx;
                    resized[((n * RESIZED + oy) * RESIZED + ox) * CHANNELS + channel] =
                        top + (bottom - top) * wy;
                }
            }
        }
    }

    let crop = (RESIZED - OUTPUT) / 2;
    let pixels = (OUTPUT * OUTPUT) as f32;
    let mut channel_sums = vec![[0.0f32; CHANNELS]; batch];
    for n in 0..batch {
        for y in 0..OUTPUT {
            for x in 0..OUTPUT {
                let source_x = crop + OUTPUT - 1 - x;
                for channel in 0..CHANNELS {
                    channel_sums[n][channel] += resized
                        [((n * RESIZED + crop + y) * RESIZED + source_x) * CHANNELS + channel]
                        + 0.05;
                }
            }
        }
    }
    let mut sum = 0.0f32;
    for n in 0..batch {
        for y in 0..OUTPUT {
            for x in 0..OUTPUT {
                let source_x = crop + OUTPUT - 1 - x;
                for channel in 0..CHANNELS {
                    let value = resized
                        [((n * RESIZED + crop + y) * RESIZED + source_x) * CHANNELS + channel]
                        + 0.05;
                    let contrasted = (value - channel_sums[n][channel] / pixels) * 1.1
                        + channel_sums[n][channel] / pixels;
                    let normalized = (contrasted - MEAN[channel]) / STDDEV[channel];
                    sum += normalized * normalized;
                }
            }
        }
    }
    sum / (batch * OUTPUT * OUTPUT * CHANNELS) as f32
}

fn tensor_pipeline(batch: usize) -> Result<TensorFunction> {
    Ok(TensorFunction::new(
        [batch as i64, INPUT as i64, INPUT as i64, CHANNELS as i64],
        DType::U8,
        |image| {
            image
                .cast(DType::F32)?
                .mul_scalar(1.0 / 255.0)?
                .resize_bilinear2d([RESIZED as i64, RESIZED as i64])?
                .center_crop([OUTPUT as i64, OUTPUT as i64])?
                .flip_left_right()?
                .adjust_brightness(0.05)?
                .adjust_contrast(1.1)?
                .normalize_nhwc(&MEAN, &STDDEV)?
                .square()?
                .mean(&[0, 1, 2, 3], false)
        },
    )?)
}

fn median_p95(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[(samples.len() - 1) * 95 / 100],
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(1..=64).contains(&args.batch)
        || !(1..=1000).contains(&args.runs)
        || !(1..=100).contains(&args.warmups)
    {
        return Err("batch/runs/warmups outside supported benchmark range".into());
    }
    let plugin = args
        .plugin
        .or_else(|| std::env::var_os("PJRT_PLUGIN_PATH").map(PathBuf::from))
        .ok_or("pass --plugin or set PJRT_PLUGIN_PATH")?;
    let data = input_data(args.batch);

    let host_expected = host_pipeline(&data, args.batch);
    let mut host_samples = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        let value = black_box(host_pipeline(black_box(&data), args.batch));
        host_samples.push(start.elapsed().as_secs_f64() * 1e3);
        if (value - host_expected).abs() > 1e-6 {
            return Err("host pipeline is not deterministic".into());
        }
    }

    let mut runtime = unsafe { Runtime::load(&plugin) }?;
    eprintln!("device={:?}", runtime.client().info()?);
    let function = tensor_pipeline(args.batch)?;
    let compile_start = Instant::now();
    let executable = function.compile(&mut runtime)?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1e3;
    let hlo = executable.optimized_hlo_proto()?;
    let hlo_text = format!("{hlo:?}");
    let fusion_count = hlo_text.matches("fusion").count();

    let host_tensor = Tensor::from_slice(
        [
            args.batch as i64,
            INPUT as i64,
            INPUT as i64,
            CHANNELS as i64,
        ],
        DType::U8,
        &data,
    )?;
    let resident = host_tensor.to_device(runtime.client())?;
    for _ in 0..args.warmups {
        let value = function.call(&mut runtime, &resident)?.to_vec::<f32>()?[0];
        // XLA may reassociate the large F32 squared-sum reduction. Pixel-level
        // interpolation/augmentation correctness is covered separately; this
        // benchmark checksum only catches gross pipeline mismatches.
        if (value - host_expected).abs() > 2e-3 {
            return Err(format!("RXLA checksum {value} differs from host {host_expected}").into());
        }
    }

    let mut resident_samples = Vec::with_capacity(args.runs);
    let mut end_to_end_samples = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        let value = function.call(&mut runtime, &resident)?.to_vec::<f32>()?[0];
        resident_samples.push(start.elapsed().as_secs_f64() * 1e3);
        black_box(value);

        let start = Instant::now();
        let input = Tensor::from_slice(
            [
                args.batch as i64,
                INPUT as i64,
                INPUT as i64,
                CHANNELS as i64,
            ],
            DType::U8,
            &data,
        )?;
        let value = function.call(&mut runtime, &input)?.to_vec::<f32>()?[0];
        end_to_end_samples.push(start.elapsed().as_secs_f64() * 1e3);
        black_box(value);
    }

    let (host_median, host_p95) = median_p95(&mut host_samples);
    let (resident_median, resident_p95) = median_p95(&mut resident_samples);
    let (end_to_end_median, end_to_end_p95) = median_p95(&mut end_to_end_samples);
    println!(
        "batch,compile_ms,host_median_ms,host_p95_ms,resident_median_ms,resident_p95_ms,end_to_end_median_ms,end_to_end_p95_ms,checksum,hlo_fusion_mentions"
    );
    println!(
        "{},{compile_ms:.3},{host_median:.3},{host_p95:.3},{resident_median:.3},{resident_p95:.3},{end_to_end_median:.3},{end_to_end_p95:.3},{host_expected:.9},{fusion_count}",
        args.batch
    );
    eprintln!(
        "Scope: deterministic U8 NHWC cast -> bilinear resize -> center crop -> flip -> brightness -> contrast -> normalize -> squared-mean. Resident excludes upload/compile and downloads one scalar; end-to-end includes Tensor copy/upload and scalar download."
    );
    Ok(())
}
