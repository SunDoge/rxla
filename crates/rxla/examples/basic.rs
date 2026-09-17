use clap::Parser;
use rxla::{DType, Runtime, Tensor};
use std::path::PathBuf;

/// Evaluate lazy tensor roots together through a caller-selected PJRT plugin.
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
    let sum = (&x + &y)?;
    let product = (&x * &y)?;

    // Loading a dynamic library executes trusted native code in this process.
    let mut runtime = unsafe { Runtime::load(arguments.plugin)? };
    let (sum, product) = runtime.eval((&sum, &product))?;

    let sums = sum.to_vec::<f32>()?;
    let products = product.to_vec::<f32>()?;
    assert_eq!(sums, [4.0, 6.0]);
    assert_eq!(products, [3.0, 8.0]);
    println!("sum={sums:?}, product={products:?}");
    Ok(())
}
