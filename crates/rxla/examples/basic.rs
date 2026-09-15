use clap::Parser;
use rxla::{DType, Runtime, Tensor};
use std::path::PathBuf;

/// Evaluate one lazy expression through a caller-selected PJRT plugin.
#[derive(Parser)]
struct Arguments {
    /// Trusted PJRT dynamic library.
    #[arg(long, env = "PJRT_PLUGIN_PATH")]
    plugin: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let x = Tensor::from_slice([2], DType::F32, [1.0, 2.0])?;
    let y = Tensor::from_slice([2], DType::F32, [3.0, 4.0])?;
    let output = (&x + &y)?;

    // Loading a dynamic library executes trusted native code in this process.
    let mut runtime = unsafe { Runtime::load(arguments.plugin)? };
    output.eval(&mut runtime)?;

    let values = output.to_vec::<f32>()?;
    assert_eq!(values, [4.0, 6.0]);
    println!("{values:?}");
    Ok(())
}
