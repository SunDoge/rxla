//! Compile and execute an offline-generated Mosaic GPU StableHLO module.
use rxla_core::{CacheLimits, Client, Compiler, DType, LoweredProgram};
use rxla_ir::TensorType;
use rxla_pjrt::NativeSupportLibrary;

fn tensor_type() -> TensorType {
    TensorType {
        dims: vec![64, 8],
        dtype: DType::F32,
        dynamic_bounds: vec![],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let [plugin, runtime, module] =
        <[String; 3]>::try_from(std::env::args().skip(1).collect::<Vec<_>>())
            .map_err(|_| "usage: mosaic_gpu CUDA_PLUGIN LIBMOSAIC_GPU_RUNTIME MOSAIC_STABLEHLO")?;
    let code = std::fs::read_to_string(module)?;
    let _runtime = unsafe { NativeSupportLibrary::load_global(runtime)? };
    let client = unsafe { Client::load(plugin)? };
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let program = LoweredProgram::from_stablehlo(
        code,
        vec![tensor_type(), tensor_type()],
        vec![tensor_type()],
    );
    let executable = compiler.compile_lowered(&program)?;
    let lhs = (0..512).map(|value| value as f32).collect::<Vec<_>>();
    let rhs = lhs.iter().map(|value| value + 1.).collect::<Vec<_>>();
    let lhs = client.buffer(&[64, 8], &lhs)?;
    let rhs = client.buffer(&[64, 8], &rhs)?;
    let actual = executable.execute(&[&lhs, &rhs])?[0].to_vec::<f32>()?;
    if actual
        .iter()
        .enumerate()
        .any(|(index, &value)| value != 2. * index as f32 + 1.)
    {
        return Err("Mosaic GPU result mismatch".into());
    }
    println!("Mosaic GPU StableHLO compiled and executed through RXLA PJRT: PASS");
    Ok(())
}
