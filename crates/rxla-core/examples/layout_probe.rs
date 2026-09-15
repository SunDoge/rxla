//! Read-only native layout diagnostics; not a zero-copy or performance test.
use rxla_core::{Client, Runtime, Tracer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let mut runtime = Runtime::new(client.clone());
    let program = Tracer::trace(|trace| {
        let x = trace.input(&[2, 3])?;
        let transposed = x.transpose(&[1, 0])?;
        let flattened = transposed.reshape(&[6])?;
        Ok(vec![transposed, flattened])
    })?;
    let input = client.buffer(&[2, 3], &[0., 1., 2., 3., 4., 5.])?;
    let mut failures = 0;
    let mut report = |label: &str, buffer: &rxla_core::Buffer| match buffer.memory_layout() {
        Ok(layout) => println!("{label}: {layout:?}"),
        Err(error) => {
            failures += 1;
            println!("{label}: QUERY FAILED: {error}");
        }
    };
    report("input", &input);
    let outputs = program.run_buffers(&mut runtime, &[&input])?;
    for (i, output) in outputs.iter().enumerate() {
        report(&format!("output {i}"), output);
        assert_eq!(output.to_vec::<f32>()?, [0., 3., 1., 4., 2., 5.]);
        println!("output {i}: shape={:?}, values=PASS", output.dimensions()?);
    }
    for shape in [vec![], vec![0, 3], vec![1, 3, 1]] {
        let count: usize = shape.iter().map(|&n| n as usize).product();
        let data = vec![2.; count];
        let buffer = client.buffer(&shape, &data)?;
        report(&format!("shape={shape:?}"), &buffer);
        let layout = buffer.memory_layout();
        assert_eq!(buffer.to_vec::<f32>()?, data);
        drop(buffer);
        println!("shape={shape:?}: owned layout after drop={layout:?}");
    }
    if failures != 0 {
        return Err(format!("{failures} layout queries failed; value checks passed").into());
    }
    Ok(())
}
