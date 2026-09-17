use rxla_core::Tensor;
use rxla_nn::{Cx, Linear, Model, ModelInput, Result};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.apply("hidden", Linear::new(256), &input)?.relu()?;
    cx.apply("head", Linear::new(10), &hidden)
}

fn main() -> Result<()> {
    let model = Model::new(apply).inputs(ModelInput::new([32, 784]));
    let applied = model.trace()?;
    let schema = applied.schema();

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", applied.prepare()?.input_count());
    Ok(())
}
