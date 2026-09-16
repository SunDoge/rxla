//! Bounded, single-thread native inference scheduling. Not a throughput benchmark.
use rxla_core::{CacheLimits, Client, Compiler, PendingExecution, Tracer};
use std::collections::VecDeque;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
struct Report {
    completed: usize,
    peak_inflight: usize,
    compilations: u64,
}

fn run(plugin: &str, requests: usize, capacity: usize) -> Result<Report> {
    if capacity == 0 {
        return Err("in-flight capacity must be positive".into());
    }
    // Keep this correctness example's integer-valued F32 reference exact and
    // host result storage bounded. These are demo limits, not tensor API limits.
    if requests > 1024 {
        return Err("example supports at most 1024 requests".into());
    }
    let client = unsafe { Client::load(plugin) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let g = Tracer::default();
    let x = g.input(&[1, 2])?;
    let w = g.input(&[2, 2])?;
    let exe = compiler.compile(&g, &x.matmul(&w)?)?;
    let weights = client.buffer(&[2, 2], &[2., 3., 4., 5.])?;
    let mut pending: VecDeque<(usize, PendingExecution)> = VecDeque::new();
    let mut results = vec![None; requests];
    let mut submitted = 0;
    let mut peak_inflight = 0;
    while submitted < requests || !pending.is_empty() {
        while submitted < requests && pending.len() < capacity {
            let value = submitted as f32;
            let input = client.buffer(&[1, 2], &[value, value + 1.])?;
            pending.push_back((submitted, exe.submit(&[&input, &weights])?));
            // `input` drops here; the pending execution retains its native owner.
            submitted += 1;
            peak_inflight = peak_inflight.max(pending.len());
        }
        // Prefer an already-completed request. If none is ready, block on the
        // oldest instead of busy-spinning. This fallback may cause head-of-line
        // blocking; it is not an event-driven executor or a fairness guarantee.
        let mut ready = 0;
        for (index, (_, execution)) in pending.iter().enumerate() {
            if execution.is_ready()? {
                ready = index;
                break;
            }
        }
        let (id, execution) = pending.remove(ready).ok_or("empty in-flight queue")?;
        let output = execution.wait()?;
        let actual = output[0].to_vec::<f32>()?;
        let value = id as f32;
        if actual != [6. * value + 4., 8. * value + 5.] {
            return Err(format!("request {id}: incorrect result {actual:?}").into());
        }
        if results[id].replace(actual).is_some() {
            return Err(format!("request {id}: duplicate completion").into());
        }
    }
    let completed = results.iter().filter(|result| result.is_some()).count();
    if completed != requests || peak_inflight > capacity || compiler.stats().misses != 1 {
        return Err("scheduler invariant failed".into());
    }
    Ok(Report {
        completed,
        peak_inflight,
        compilations: compiler.stats().misses,
    })
}

fn main() -> Result<()> {
    let report = run(&std::env::var("PJRT_PLUGIN_PATH")?, 17, 3)?;
    println!(
        "{} requests verified; peak retained tasks {}; compilations {}",
        report.completed, report.peak_inflight, report.compilations
    );
    println!("Single-thread correctness only; no kernel overlap or speedup claim.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_limits_rejected_before_plugin_load() {
        assert!(
            run("/missing/plugin", 17, 0)
                .unwrap_err()
                .to_string()
                .contains("capacity")
        );
        assert!(
            run("/missing/plugin", 1025, 3)
                .unwrap_err()
                .to_string()
                .contains("1024")
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn real_bounded_inference() -> Result<()> {
        let plugin = std::env::var("PJRT_PLUGIN_PATH")?;
        // Sequential clients, including a partial final window and fewer
        // requests than slots. Never create two default CUDA allocators at once.
        for (requests, capacity) in [(0, 3), (17, 1), (17, 3), (2, 4)] {
            let report = run(&plugin, requests, capacity)?;
            assert_eq!(report.completed, requests);
            assert_eq!(report.peak_inflight, requests.min(capacity));
            assert_eq!(report.compilations, 1);
        }
        Ok(())
    }
}
