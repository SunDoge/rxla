//! Host-to-host latency: identical work on CPU, GPU, or CPU -> GPU -> CPU.
//! Resident weights, synchronous stages, no model/performance portability claim.
use rxla_core::{Client, Tracer};
use std::time::Instant;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(100);
    if !(1..=10000).contains(&rounds) || args.next().is_some() {
        return Err("usage: heterogeneous_bench [ROUNDS:1..=10000]".into());
    }
    let cpu = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH")?) }?;
    let gpu = unsafe { Client::load(std::env::var("PJRT_CUDA_PLUGIN_PATH")?) }?;
    if !cpu.info()?.platform.eq_ignore_ascii_case("cpu")
        || !gpu.info()?.platform.eq_ignore_ascii_case("cuda")
    {
        return Err("expected CPU and CUDA plugins".into());
    }
    eprintln!(
        "CPU={:?}; GPU={:?}; debug={}",
        cpu.info()?,
        gpu.info()?,
        cfg!(debug_assertions)
    );
    let names = ["cpu_whole", "gpu_whole", "cpu_gpu_cpu"];
    let mut rows = Vec::new();
    for (batch, width) in [(1usize, 64usize), (8, 256), (32, 512)] {
        let shape = [batch as i64, width as i64];
        let wshape = [width as i64, width as i64];
        let g = Tracer::default();
        let x = g.input(&shape)?;
        let w = g.input(&wshape)?;
        let output = x
            .mul_scalar(0.5)?
            .add_scalar(1.)?
            .matmul(&w)?
            .sum(&[1], false)?;
        let whole_cpu = g.compile(&cpu, &output)?;
        let whole_gpu = g.compile(&gpu, &output)?;
        let pre_g = Tracer::default();
        let x = pre_g.input(&shape)?;
        let pre = pre_g.compile(&cpu, &x.mul_scalar(0.5)?.add_scalar(1.)?)?;
        let infer_g = Tracer::default();
        let x = infer_g.input(&shape)?;
        let w = infer_g.input(&wshape)?;
        let infer = infer_g.compile(&gpu, &x.matmul(&w)?)?;
        let post_g = Tracer::default();
        let x = post_g.input(&shape)?;
        let post = post_g.compile(&cpu, &x.sum(&[1], false)?)?;
        let input: Vec<_> = (0..batch * width)
            .map(|i| (i as i32 % 7 - 3) as f32 / 8.)
            .collect();
        let weights: Vec<_> = (0..width * width)
            .map(|i| (i as i32 % 5 - 2) as f32 / 64.)
            .collect();
        let cpu_w = cpu.buffer(&wshape, &weights)?;
        let gpu_w = gpu.buffer(&wshape, &weights)?;
        let reference: Vec<f64> = (0..batch)
            .map(|b| {
                (0..width)
                    .map(|j| {
                        (0..width)
                            .map(|k| {
                                (input[b * width + k] as f64 * 0.5 + 1.)
                                    * weights[k * width + j] as f64
                            })
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        let execute = |mode| -> Result<Vec<f32>> {
            let result = match mode {
                0 => whole_cpu.execute(&[&cpu.buffer(&shape, &input)?, &cpu_w])?,
                1 => whole_gpu.execute(&[&gpu.buffer(&shape, &input)?, &gpu_w])?,
                2 => {
                    let prepared = pre.execute(&[&cpu.buffer(&shape, &input)?])?;
                    let remote = prepared[0].copy_to_client_via_host(&gpu)?;
                    let output = infer.execute(&[&remote, &gpu_w])?;
                    post.execute(&[&output[0].copy_to_client_via_host(&cpu)?])?
                }
                _ => unreachable!(),
            };
            Ok(result[0].to_vec::<f32>()?)
        };
        let check = |output: &[f32]| -> Result<()> {
            if output.len() != reference.len()
                || output
                    .iter()
                    .zip(&reference)
                    .any(|(&a, &b)| !a.is_finite() || (a as f64 - b).abs() > 1e-5)
            {
                return Err("output differs from independent F64 reference".into());
            }
            Ok(())
        };
        for _ in 0..10 {
            for mode in 0..3 {
                check(&execute(mode)?)?;
            }
        }
        let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for round in 0..rounds {
            for mode in [0, 1, 2, 2, 1, 0] {
                let start = Instant::now();
                let output = execute(mode)?;
                let micros = start.elapsed().as_secs_f64() * 1e6;
                check(&output)?;
                samples[mode].push(micros);
                rows.push((batch, width, round, names[mode], micros));
            }
        }
        for (mode, samples) in samples.iter_mut().enumerate() {
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "batch={batch} width={width} {} n={} median_us={:.3} p95_us={:.3}",
                names[mode],
                samples.len(),
                samples[samples.len() / 2],
                samples[(samples.len() - 1) * 95 / 100]
            );
        }
    }
    println!("batch,width,round,mode,microseconds");
    for (batch, width, round, mode, micros) in rows {
        println!("{batch},{width},{round},{mode},{micros:.3}");
    }
    eprintln!(
        "Includes input upload, stage execution/copies and final download; excludes compilation and resident weight upload. Not model throughput or overlap evidence."
    );
    Ok(())
}
