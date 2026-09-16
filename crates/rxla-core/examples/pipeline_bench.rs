//! Same-client synthetic comparison: fused graph, split synchronous calls, and
//! split submissions. Includes final host download; excludes upload/compilation.
use rxla_core::{Buffer, Client, Executable, Tracer};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const BATCH: usize = 8;
const WIDTH: usize = 64;
const NAMES: [&str; 3] = ["fused", "split_sync", "split_submit"];

fn reference(x: &[f32], w: &[f32]) -> Vec<f64> {
    let multiply = |x: &[f64]| {
        (0..BATCH * WIDTH)
            .map(|i| {
                (0..WIDTH)
                    .map(|k| x[i / WIDTH * WIDTH + k] * w[k * WIDTH + i % WIDTH] as f64)
                    .sum()
            })
            .collect::<Vec<f64>>()
    };
    multiply(&multiply(&x.iter().map(|&v| v as f64).collect::<Vec<_>>()))
}

fn execute(
    mode: usize,
    stage: &Executable,
    fused: &Executable,
    x: &Buffer,
    w: &Buffer,
) -> Result<Vec<f32>> {
    let output = match mode {
        0 => fused.execute(&[x, w])?,
        1 => {
            let first = stage.execute(&[x, w])?;
            stage.execute(&[&first[0], w])?
        }
        2 => {
            let first = stage.submit(&[x, w])?;
            let second = stage.submit(&[&first.outputs()[0], w])?;
            let output = second.wait()?;
            first.wait()?;
            output
        }
        _ => return Err("invalid benchmark mode".into()),
    };
    Ok(output[0].to_vec::<f32>()?)
}

fn check(output: &[f32], reference: &[f64]) -> Result<()> {
    if output.len() != reference.len()
        || output
            .iter()
            .zip(reference)
            .any(|(&a, &b)| !a.is_finite() || (a as f64 - b).abs() > 1e-5)
    {
        return Err("output differs from independent F64 reference".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(100);
    if !(1..=10000).contains(&rounds) || args.next().is_some() {
        return Err("usage: pipeline_bench [ROUNDS:1..=10000] > NEW.csv".into());
    }
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    eprintln!(
        "Device: {:?}; debug_assertions={}",
        client.info()?,
        cfg!(debug_assertions)
    );
    let g = Tracer::default();
    let x = g.input(&[BATCH as i64, WIDTH as i64])?;
    let w = g.input(&[WIDTH as i64, WIDTH as i64])?;
    let first = x.matmul(&w)?;
    let stage = g.compile(&client, &first)?;
    let fused = g.compile(&client, &first.matmul(&w)?)?;
    let host_x: Vec<_> = (0..BATCH * WIDTH)
        .map(|i| (i as i32 % 7 - 3) as f32 / 8.)
        .collect();
    let host_w: Vec<_> = (0..WIDTH * WIDTH)
        .map(|i| (i as i32 % 5 - 2) as f32 / 64.)
        .collect();
    let reference = reference(&host_x, &host_w);
    let x = client.buffer(&[BATCH as i64, WIDTH as i64], &host_x)?;
    let w = client.buffer(&[WIDTH as i64, WIDTH as i64], &host_w)?;
    for _ in 0..10 {
        for mode in 0..3 {
            check(&execute(mode, &stage, &fused, &x, &w)?, &reference)?;
        }
    }
    let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(rounds * 2));
    // Buffer CSV rows until all measurements finish; printing cannot perturb
    // individual measurements or favor one mode's position within a round.
    let mut rows = Vec::with_capacity(rounds * 6);
    for round in 0..rounds {
        for mode in [0, 1, 2, 2, 1, 0] {
            let start = Instant::now();
            let output = execute(mode, &stage, &fused, &x, &w)?;
            let micros = start.elapsed().as_secs_f64() * 1e6;
            check(&output, &reference)?;
            samples[mode].push(micros);
            rows.push((round, mode, micros));
        }
    }
    println!("round,mode,microseconds");
    for (round, mode, micros) in rows {
        println!("{round},{},{micros:.3}", NAMES[mode]);
    }
    for (mode, values) in samples.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        eprintln!(
            "{}: n={} median_us={:.3} p95_us={:.3}",
            NAMES[mode],
            values.len(),
            values[values.len() / 2],
            values[(values.len() - 1) * 95 / 100]
        );
    }
    eprintln!(
        "All outputs checked; synthetic fixed-shape latency, not model throughput or kernel-overlap evidence."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_reference_identity_and_validation() {
        let x = vec![1.; BATCH * WIDTH];
        let mut w = vec![0.; WIDTH * WIDTH];
        for i in 0..WIDTH {
            w[i * WIDTH + i] = 1.;
        }
        assert_eq!(reference(&x, &w), vec![1.; BATCH * WIDTH]);
        assert!(check(&x, &reference(&x, &w)).is_ok());
        assert!(check(&[f32::NAN], &[0.]).is_err());
        assert!(check(&[], &[0.]).is_err());
    }
}
