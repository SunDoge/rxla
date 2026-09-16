//! Synthetic sampling microbenchmark, not LLM decode throughput.
use rxla_core::{
    Client, Tracer,
    random::{top_p_categorical_from_bits, topk_categorical_from_bits},
};
use std::{io::Write, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const VOCAB: usize = 32000;
const TOP_K: usize = 40;
const TOP_P: f32 = 0.9;

fn reference(logits: &[f32], words: &[i32], nucleus: bool) -> usize {
    let mut order: Vec<_> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    if nucleus {
        let maximum = logits[order[0]] as f64;
        let weights: Vec<_> = order
            .iter()
            .map(|&i| (logits[i] as f64 - maximum).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut mass = 0.;
        let mut count = 0;
        for weight in weights {
            count += 1;
            mass += weight / total;
            if mass >= TOP_P as f64 {
                break;
            }
        }
        order.truncate(count);
    } else {
        order.truncate(TOP_K);
    }
    let maximum = logits[order[0]] as f64;
    let score = |rank: usize| {
        let u = (((words[rank] as u32) >> 9) as f64 + 0.5) / 8_388_608.;
        logits[order[rank]] as f64 - maximum - (-u.ln()).ln()
    };
    let winner = (0..order.len())
        .max_by(|&a, &b| score(a).total_cmp(&score(b)).then_with(|| b.cmp(&a)))
        .unwrap();
    order[winner]
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().map(|x| x.parse()).transpose()?.unwrap_or(100);
    let mut samples_path = None;
    let mut valid_first = false;
    let mut reuse_host = false;
    while let Some(flag) = args.next() {
        if flag == "--samples" && samples_path.is_none() {
            samples_path = Some(std::path::PathBuf::from(
                args.next().ok_or("missing samples path")?,
            ));
        } else if flag == "--valid-first" && !valid_first {
            valid_first = true;
        } else if flag == "--reuse-host" && !reuse_host {
            reuse_host = true;
        } else {
            return Err("unexpected or duplicate benchmark option".into());
        }
    }
    if !(1..=10000).contains(&rounds) {
        return Err(
            "usage: sample_vocabulary [ABBA_ROUNDS:1..=10000 [--samples NEW.csv] [--valid-first] [--reuse-host]]"
                .into(),
        );
    }
    if samples_path.as_ref().is_some_and(|path| path.exists()) {
        return Err("samples output already exists".into());
    }
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    eprintln!("Device: {:?}", client.info()?);
    // Fixed finite inputs, not a statistical RNG-quality experiment. Supplying
    // words as parameters prevents compiler constant-folding the sampler.
    let logits: Vec<_> = (0..VOCAB)
        .map(|i| -((i * 7919 % VOCAB) as f32) / 512.)
        .collect();
    let words: Vec<_> = (0..VOCAB)
        .map(|i| (i as u32).wrapping_mul(0x9e3779b9).rotate_left(13) as i32)
        .collect();
    let values = client.buffer(&[VOCAB as i64], &logits)?;
    let mut plans = Vec::new();
    for nucleus in [false, true] {
        let g = Tracer::default();
        let x = g.input(&[VOCAB as i64])?;
        let count = if nucleus { VOCAB } else { TOP_K };
        let bits = g.input_i32(&[count as i64])?;
        let sample = if nucleus {
            top_p_categorical_from_bits(&x, &bits, TOP_P, 0)?
        } else {
            topk_categorical_from_bits(&x, &bits, TOP_K, 0)?
        };
        let start = Instant::now();
        let executable = g.compile_many(&client, &[sample.indices, sample.valid])?;
        let compile_ms = start.elapsed().as_secs_f64() * 1000.;
        let bits = client.buffer(&[count as i64], &words[..count])?;
        let expected = reference(&logits, &words, nucleus) as i32;
        plans.push((executable, bits, expected, compile_ms));
    }
    // Persist across warmups and measured calls; native output buffers are still
    // newly returned by execute. Only host scalar output storage is reused.
    let mut token_storage = [0i32];
    let mut valid_storage = [0f32];
    let mut run = |mode: usize| -> Result<[f64; 6]> {
        let (executable, bits, expected, _) = &plans[mode];
        let start = Instant::now();
        let outputs = executable.execute(&[&values, bits])?;
        let executed_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut token_matches = || -> Result<bool> {
            if reuse_host {
                outputs[0].copy_to(&mut token_storage)?;
                Ok(token_storage == [*expected])
            } else {
                Ok(outputs[0].to_vec::<i32>()? == [*expected])
            }
        };
        let mut valid_matches = || -> Result<bool> {
            if reuse_host {
                outputs[1].copy_to(&mut valid_storage)?;
                Ok(valid_storage == [1.])
            } else {
                Ok(outputs[1].to_vec::<f32>()? == [1.])
            }
        };
        let (token_matches, valid_matches, token_cost, valid_cost) = if valid_first {
            let valid = valid_matches()?;
            let first_ms = start.elapsed().as_secs_f64() * 1000.;
            let token = token_matches()?;
            let second_ms = start.elapsed().as_secs_f64() * 1000.;
            (token, valid, second_ms - first_ms, first_ms - executed_ms)
        } else {
            let token = token_matches()?;
            let first_ms = start.elapsed().as_secs_f64() * 1000.;
            let valid = valid_matches()?;
            let second_ms = start.elapsed().as_secs_f64() * 1000.;
            (token, valid, first_ms - executed_ms, second_ms - first_ms)
        };
        if !token_matches || !valid_matches {
            return Err("device sample disagrees with independent F64 reference".into());
        }
        let downloaded_ms = start.elapsed().as_secs_f64() * 1000.;
        drop(outputs);
        let total_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok([
            total_ms,
            executed_ms,
            downloaded_ms - executed_ms,
            total_ms - downloaded_ms,
            token_cost,
            valid_cost,
        ])
    };
    for _ in 0..5 {
        run(0)?;
        run(1)?;
    }
    let mut samples = [Vec::new(), Vec::new()];
    let mut records = Vec::with_capacity(rounds * 4);
    for _ in 0..rounds {
        for mode in [0, 1, 1, 0] {
            let elapsed = run(mode)?;
            samples[mode].push(elapsed[0]);
            records.push((mode, elapsed));
        }
    }
    // File creation/writes are outside measured calls; never overwrite an old
    // run. Execution/validation failures precede table creation; an I/O error
    // during writing may leave a partial diagnostic file.
    if let Some(path) = samples_path {
        let mut writer = std::io::BufWriter::new(std::fs::File::create_new(path)?);
        writeln!(
            writer,
            "sequence,abba_round,abba_slot,mode,total_ms,execute_wait_ms,download_validate_ms,cleanup_ms,token_download_check_ms,valid_download_check_ms,download_order,host_storage"
        )?;
        for (sequence, &(mode, elapsed)) in records.iter().enumerate() {
            writeln!(
                writer,
                "{sequence},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
                sequence / 4,
                sequence % 4,
                if mode == 0 { "top_k_40" } else { "top_p_0.9" },
                elapsed[0],
                elapsed[1],
                elapsed[2],
                elapsed[3],
                elapsed[4],
                elapsed[5],
                if valid_first {
                    "valid_then_token"
                } else {
                    "token_then_valid"
                },
                if reuse_host {
                    "reused_slice"
                } else {
                    "new_vec"
                }
            )?;
        }
        writer.flush()?;
    }
    println!("mode,vocab,samples,compile_ms,mean_ms,median_ms,min_ms,max_ms,token,p95_ms,p99_ms");
    for (mode, times) in samples.iter_mut().enumerate() {
        times.sort_by(f64::total_cmp);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let median = (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.;
        println!(
            "{},{VOCAB},{},{:.6},{mean:.6},{median:.6},{:.6},{:.6},{},{:.6},{:.6}",
            if mode == 0 { "top_k_40" } else { "top_p_0.9" },
            times.len(),
            plans[mode].3,
            times[0],
            times[times.len() - 1],
            plans[mode].2,
            percentile(times, 95),
            percentile(times, 99)
        );
    }
    eprintln!(
        "Scope: fixed synthetic logits/words; 5 warmups per mode, ABBA interleaving; resident input buffers. Timings include synchronous execution, two scalar downloads, validation and output cleanup. Exclude graph build, compilation, uploads, RNG generation and LLM computation. Not pure kernel timings or decoding throughput."
    );
    eprintln!(
        "Download order: {}",
        if valid_first {
            "valid then token"
        } else {
            "token then valid"
        }
    );
    eprintln!(
        "Host storage: {}",
        if reuse_host {
            "reused slice"
        } else {
            "new Vec"
        }
    );
    Ok(())
}

// Nearest-rank sample percentile, not an estimate or service-level guarantee.
fn percentile(sorted: &[f64], percent: usize) -> f64 {
    sorted[(sorted.len() * percent).div_ceil(100) - 1]
}

#[cfg(test)]
mod tests {
    #[test]
    fn percentiles_use_nearest_rank() {
        assert_eq!(super::percentile(&[3.], 99), 3.);
        let values: Vec<_> = (1..=200).map(f64::from).collect();
        assert_eq!(super::percentile(&values, 95), 190.);
        assert_eq!(super::percentile(&values, 99), 198.);
    }
    #[test]
    fn reference_preserves_first_index_for_tied_scores() {
        for nucleus in [false, true] {
            assert_eq!(super::reference(&[0.; 40], &[0; 40], nucleus), 0);
        }
    }
}
