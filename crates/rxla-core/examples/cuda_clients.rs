//! CUDA-plugin-specific options; requires a trusted CUDA PJRT plugin.
//! Two simultaneously live clients, sequential execution, not a throughput test.
use rxla_core::{Client, ClientOptions, Runtime, Tracer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = std::env::var("PJRT_PLUGIN_PATH")?;
    let options = ClientOptions::new()
        .set("allocator", "bfc")
        .set("preallocate", false)
        .set("memory_fraction", 0.25_f32);
    // The operator explicitly trusts this plugin and its option ABI.
    let clients = [
        unsafe { Client::load_with_options(&plugin, &options)? },
        unsafe { Client::load_with_options(&plugin, &options)? },
    ];
    let program = Tracer::trace(|trace| {
        let input = trace.input(&[2, 2])?;
        Ok(vec![input.matmul(&input)?])
    })?;
    let mut programs = Vec::new();
    for client in clients {
        let info = client.info()?;
        assert_eq!(info.platform.to_ascii_lowercase(), "cuda");
        println!("{info:?}");
        let mut runtime = Runtime::new(client);
        let executable = program.compile(&mut runtime)?;
        programs.push((runtime, executable));
    }
    for _ in 0..3 {
        for (_, executable) in &programs {
            assert_eq!(executable.run(&[&[1., 2., 3., 4.]])?, [7., 10., 15., 22.]);
        }
    }
    println!("Two live CUDA clients: 6 executions passed, BFC growth enabled, fraction 0.25 each");
    Ok(())
}
