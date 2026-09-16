use rxla_core::Tensor;
use rxla_nn::{Cx, Model, ModelInput, Result};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.linear("hidden", &input, 256)?.relu()?;
    cx.linear("head", &hidden, 10)
}

fn main() -> Result<()> {
    let model = Model::new(apply).inputs(ModelInput::new([32, 784]));
    let (schema, applied) = model.trace()?;

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", applied.prepare()?.input_count());
    Ok(())
}
