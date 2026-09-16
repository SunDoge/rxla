//! Synthetic prefix benchmark: resident inputs, synchronous execute + download.
use rxla_core::{Client, Tracer};
use std::time::Instant;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn reference(values: &[f32], length: usize) -> Vec<f64> {
    values
        .chunks(length)
        .flat_map(|row| {
            let mut sum = f64::NEG_INFINITY;
            row.iter().map(move |&value| {
                let value = value as f64;
                sum = sum.max(value) + (-(sum - value).abs()).exp().ln_1p();
                sum
            })
        })
        .collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(30);
    if !(5..=10000).contains(&rounds) || args.next().is_some() {
        return Err("usage: logcumsumexp_bench [ROUNDS:5..=10000]".into());
    }
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    eprintln!(
        "device={:?}; debug_assertions={}; rounds={rounds}",
        client.info()?,
        cfg!(debug_assertions)
    );
    println!("length,mode,compile_ms,median_ms,p90_ms,max_abs_error");
    for length in [64, 256, 1024, 4096] {
        let graph = Tracer::default();
        let x = graph.input(&[4, length as i64])?;
        let scan = x.logcumsumexp_tree(1)?;
        let doubling = x.logcumsumexp(1)?;
        // This comparator is deliberately only used on bounded inputs: it is
        // not a stable replacement for the public operator on arbitrary data.
        let direct = x.exp()?.cumsum(1)?.log()?;
        let started = Instant::now();
        let scan = graph.compile(&client, &scan)?;
        let scan_compile = started.elapsed().as_secs_f64() * 1000.;
        let started = Instant::now();
        let direct = graph.compile(&client, &direct)?;
        let direct_compile = started.elapsed().as_secs_f64() * 1000.;
        let started = Instant::now();
        let doubling = graph.compile(&client, &doubling)?;
        let doubling_compile = started.elapsed().as_secs_f64() * 1000.;
        let safe: Vec<_> = (0..4 * length)
            .map(|i| ((i * 37 % 101) as f32 - 50.) / 12.5)
            .collect();
        let wide: Vec<_> = (0..4 * length)
            .map(|i| -1000. + 2000. * (i % length) as f32 / (length - 1) as f32)
            .collect();
        let references = [
            reference(&safe, length),
            reference(&safe, length),
            reference(&wide, length),
            reference(&safe, length),
            reference(&wide, length),
        ];
        let safe = client.buffer(&[4, length as i64], &safe)?;
        let wide = client.buffer(&[4, length as i64], &wide)?;
        let modes = [
            (&scan, &safe),
            (&direct, &safe),
            (&scan, &wide),
            (&doubling, &safe),
            (&doubling, &wide),
        ];
        let mut times: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        let mut errors = [0_f64; 5];
        for round in 0..rounds + 3 {
            // Rotate ordering; warm up each path three times before measurement.
            for offset in 0..5 {
                let index = (round + offset) % 5;
                let (executable, input) = modes[index];
                let started = Instant::now();
                let actual = executable.execute(&[input])?[0].to_vec::<f32>()?;
                let elapsed = started.elapsed().as_secs_f64() * 1000.;
                if round >= 3 {
                    times[index].push(elapsed);
                }
                if actual.len() != references[index].len() {
                    return Err("output length mismatch".into());
                }
                for (&a, &b) in actual.iter().zip(&references[index]) {
                    if !a.is_finite() || (a as f64 - b).abs() > 0.002 {
                        return Err(format!("length={length} mode={index}: {a} vs {b}").into());
                    }
                    errors[index] = errors[index].max((a as f64 - b).abs());
                }
            }
        }
        for (index, name) in [
            "tree_bounded",
            "direct_bounded",
            "tree_wide",
            "doubling_bounded",
            "doubling_wide",
        ]
        .into_iter()
        .enumerate()
        {
            times[index].sort_by(f64::total_cmp);
            let compile = if index >= 3 {
                doubling_compile
            } else if index == 1 {
                direct_compile
            } else {
                scan_compile
            };
            println!(
                "{length},{name},{compile:.6},{:.6},{:.6},{:.9}",
                times[index][rounds / 2],
                times[index][rounds * 9 / 10],
                errors[index]
            );
        }
    }
    Ok(())
}

#[test]
fn stable_reference_handles_wide_prefixes() {
    let values = reference(&[-1000., 1000., 1000., -1000.], 2);
    assert_eq!(values, [-1000., 1000., 1000., 1000.]);
}
