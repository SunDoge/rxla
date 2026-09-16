use rxla_core::{Client, Runtime, Tracer};
use std::{sync::Arc, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Loading native code requires trusting the operator-selected plugin.
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    let mut runtime = Runtime::new(client)?;
    // Weights are runtime inputs, so changing their values does not change the key.
    let program = Tracer::trace(|trace| {
        let x = trace.input(&[2, 3])?;
        let w = trace.input(&[3, 2])?;
        Ok(vec![x.matmul(&w)?.silu()?])
    })?;
    let start = Instant::now();
    let executable = program.compile(&mut runtime)?;
    let cold = start.elapsed();
    let compile_time = runtime.stats().compile_time;
    let start = Instant::now();
    for _ in 0..1000 {
        let cached = program.compile(&mut runtime)?;
        assert!(Arc::ptr_eq(&executable, &cached));
    }
    let warm = start.elapsed() / 1000;
    assert_eq!(runtime.stats().compile_time, compile_time);
    assert_eq!((runtime.stats().misses, runtime.stats().hits), (1, 1000));
    // Reuse one compiled shape while changing both data and weights. Check
    // against host F64 arithmetic, not a second execution of the same graph.
    for shift in [-2., -0.5, 0., 2.] {
        let values = [1., 2., 3., 4., 5., 6.].map(|v| v + shift);
        let weights = [1., 0., 0., 1., 1., 1.].map(|v| v - shift * 0.25);
        let output = executable.run(&[&values, &weights])?;
        for row in 0..2 {
            for column in 0..2 {
                let dot: f64 = (0..3)
                    .map(|k| values[row * 3 + k] as f64 * weights[k * 2 + column] as f64)
                    .sum();
                let expected = dot / (1. + (-dot).exp());
                assert!((output[row * 2 + column] as f64 - expected).abs() < 2e-6);
            }
        }
    }
    let result = executable.run(&[&[1., 2., 3., 4., 5., 6.], &[1., 0., 0., 1., 1., 1.]])?;
    println!("Output: {result:?}");
    println!("First compilation: {cold:?}; average prepared Program cache lookup: {warm:?}");
    println!("Cache-miss compilation attempt time (excludes lowering/cache I/O): {compile_time:?}");
    println!("Cache: {:?}", runtime.stats());
    // Clear releases only cache references. The executable still owns its resources.
    runtime.clear();
    drop(runtime);
    assert_eq!(
        executable.run(&[&[1., 2., 3., 4., 5., 6.], &[1., 0., 0., 1., 1., 1.]])?,
        result
    );
    Ok(())
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cached_runtime_weights_and_timing() {
    main().unwrap();
}
