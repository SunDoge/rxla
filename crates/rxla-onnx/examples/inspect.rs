use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
struct Args {
    model: PathBuf,
    /// Concrete dimensions for the model's first input, for example 1,3,640,960.
    #[arg(long, value_delimiter = ',')]
    shape: Option<Vec<usize>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = if let Some(shape) = args.shape {
        let unspecialized = rxla_onnx::Model::load(&args.model)?.summary();
        let input = unspecialized.inputs.first().ok_or("model has no inputs")?;
        rxla_onnx::Model::load_specialized(
            &args.model,
            &[rxla_onnx::InputSpec {
                name: input.name.clone(),
                shape,
            }],
        )?
    } else {
        rxla_onnx::Model::load(&args.model)?
    };
    let summary = model.summary();
    let shapes = model.infer_shapes()?;
    println!("nodes: {}", summary.node_count);
    println!("inputs: {:?}", summary.inputs);
    println!("outputs: {:?}", summary.outputs);
    println!("inferred outputs: {:?}", shapes.outputs);
    println!(
        "declared shape mismatches: {}",
        shapes.declared_mismatches.len()
    );
    for (operator, count) in summary.operators {
        println!("{operator}: {count}");
    }
    let mut imported = model.import()?;
    println!("RXLA parameters: {}", imported.parameters.len());
    println!("RXLA outputs: {}", imported.outputs.len());
    let stablehlo = imported
        .program
        .stablehlo_program(&imported.outputs, true)?;
    println!("StableHLO bytes: {}", stablehlo.code.len());
    Ok(())
}
