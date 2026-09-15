use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Parser;
use rxla_core::{CacheLimits, Client, Compiler, LoweredProgram};
use rxla_safetensors::SafeTensors;

#[derive(Parser)]
struct Args {
    onnx: PathBuf,
    bundle: PathBuf,
    #[arg(long, default_value_t = 10)]
    runs: usize,
    /// Export the exact StableHLO ABI and F32 arguments for backend A/B tests.
    #[arg(long)]
    export_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = rxla_onnx::Model::load_specialized(
        &args.onnx,
        &[rxla_onnx::InputSpec {
            name: "x".into(),
            shape: vec![1, 3, 640, 960],
        }],
    )?;
    let mut imported = model.import()?;
    let stablehlo = imported
        .program
        .stablehlo_program(&imported.outputs, true)?;
    let stablehlo_code = stablehlo.code.clone();
    let lowered =
        LoweredProgram::from_stablehlo(stablehlo.code, stablehlo.inputs, stablehlo.outputs);

    let plugin = std::env::var("PJRT_PLUGIN_PATH")?;
    let client = unsafe { Client::load(plugin)? };
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let started = Instant::now();
    let executable = compiler.compile_lowered(&lowered)?;
    let compile_ms = started.elapsed().as_secs_f64() * 1e3;

    let mut checkpoint = SafeTensors::open(args.bundle.join("tensors.safetensors"))?;
    let input = checkpoint.read_f32("input/0")?;
    let expected = checkpoint.read_f32("expected/0")?;
    if let Some(directory) = &args.export_dir {
        export_jax_fixture(
            directory,
            stablehlo_code.as_bytes(),
            &input.shape,
            &input.values,
            &expected.values,
            &imported.parameters,
        )?;
    }
    let input = client.buffer(&input.shape, &input.values)?;
    let mut buffers = vec![input];
    for parameter in &imported.parameters {
        let shape: Vec<_> = parameter.shape.iter().map(|&dim| dim as i64).collect();
        buffers.push(client.buffer(&shape, &parameter.values)?);
    }
    let refs = buffers.iter().collect::<Vec<_>>();
    let warmup = executable.execute(&refs)?;
    let warmup_values = warmup[0].to_vec::<f32>()?;
    let mut max_error = 0.0_f32;
    for (&actual, &reference) in warmup_values.iter().zip(&expected.values) {
        max_error = max_error.max((actual - reference).abs());
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let started = Instant::now();
        let outputs = executable.execute(&refs)?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
        // Materialize after timing while retaining each output until completion.
        let _ = outputs[0].to_vec::<f32>()?;
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!("compile_ms={compile_ms:.3}");
    println!("execute_mean_ms={mean:.3}");
    println!("max_abs_error={max_error:.9}");
    println!("parameters={}", imported.parameters.len());
    Ok(())
}

fn export_jax_fixture(
    directory: &Path,
    stablehlo: &[u8],
    input_shape: &[i64],
    input: &[f32],
    expected: &[f32],
    parameters: &[rxla_onnx::Parameter],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(directory)?;
    fs::write(directory.join("module.mlir"), stablehlo)?;
    write_f32(directory.join("arg-0000.f32"), input)?;
    write_f32(directory.join("expected.f32"), expected)?;
    let mut manifest = BufWriter::new(File::create(directory.join("arguments.txt"))?);
    writeln!(manifest, "{}", comma_shape(input_shape))?;
    for (index, parameter) in parameters.iter().enumerate() {
        write_f32(
            directory.join(format!("arg-{:04}.f32", index + 1)),
            &parameter.values,
        )?;
        let shape = parameter
            .shape
            .iter()
            .map(|&dim| dim as i64)
            .collect::<Vec<_>>();
        writeln!(manifest, "{}", comma_shape(&shape))?;
    }
    Ok(())
}

fn comma_shape(shape: &[i64]) -> String {
    shape
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn write_f32(path: PathBuf, values: &[f32]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()
}
