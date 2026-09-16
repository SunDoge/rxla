use rxla_core::{Client, Runtime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("PJRT_PLUGIN_PATH")?;
    println!("Loading operator-selected native plugin: {path}");
    // Metadata queries do not sandbox a plugin: it must already be trusted.
    let client = unsafe { Client::load(&path)? };
    println!("{:#?}", client.info()?);
    let mut runtime = Runtime::new(client)?;
    let x = Tensor::from_slice([2], rxla_core::DType::F32, [3., -2.])?;
    let y = x.mul_scalar(2.)?.add_scalar(1.)?;
    let result = y
        .eval(&mut runtime)?
        .to_buffer(runtime.client())?
        .to_vec::<f32>()?;
    if result != [7., -3.] {
        return Err(format!("HLO execution mismatch: {result:?}").into());
    }
    println!("Tensor-first Pliron compilation, transfer and execution: PASS");
    Ok(())
}
