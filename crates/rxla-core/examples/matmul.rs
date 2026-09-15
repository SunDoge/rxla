use rxla_core::{Client, Runtime, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = std::env::var("PJRT_PLUGIN_PATH")?;
    // Explicitly trust the plugin selected by the operator.
    let client = unsafe { Client::load(plugin)? };
    let info = client.info()?;
    println!(
        "PJRT platform: {}; devices: {:?}",
        info.platform, info.addressable_devices
    );
    let mut runtime = Runtime::new(client);
    let x = Tensor::from_slice([2, 3], rxla_core::DType::F32, [1., 2., 3., 4., 5., 6.])?;
    let w = Tensor::from_slice([3, 2], rxla_core::DType::F32, [1., 2., 3., 4., 5., 6.])?;
    let bias = Tensor::from_slice([2, 2], rxla_core::DType::F32, [1., 1., 1., 1.])?;
    let y = x.matmul(&w)?.add(&bias)?;
    let y = y.eval(&mut runtime)?;
    let actual = y.to_buffer(runtime.client())?.to_vec::<f32>()?;
    assert_eq!(actual, [23., 29., 50., 65.]);
    println!("Tensor → Pliron → StableHLO → PJRT: {actual:?}");
    Ok(())
}
