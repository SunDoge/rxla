//! Explicit native artifact export/load. Not an automatic persistent cache.
use rxla_core::{Client, Tracer};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 || (args[0] != "save" && args[0] != "load") {
        return Err("usage: artifact <save|load> <trusted-artifact-path>".into());
    }
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    if args[0] == "save" {
        let graph = Tracer::default();
        let x = graph.input(&[2, 3])?;
        let w = graph.input(&[3, 2])?;
        let y = x.matmul(&w)?;
        let sum = y.sum(&[1], false)?;
        let bytes = graph.compile_many(&client, &[y, sum])?.serialize()?;
        // Never overwrite an existing artifact. The operator owns distribution,
        // plugin pinning and protection against untrusted file modification.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[1])?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        println!("Saved {} bytes of native executable", bytes.len());
    } else {
        let bytes = std::fs::read(&args[1])?;
        // Only use artifacts created by this example with the same trusted plugin
        // and compatible hardware. Never load artifacts from untrusted users.
        let executable = unsafe { client.deserialize_executable(&bytes)? };
        let x = client.buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.])?;
        let w = client.buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.])?;
        let output = executable.execute(&[&x, &w])?;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].to_vec::<f32>()?, [22., 28., 49., 64.]);
        assert_eq!(output[1].to_vec::<f32>()?, [50., 113.]);
        println!("Loaded and executed native artifact: PASS (no HLO compile call)");
    }
    Ok(())
}
