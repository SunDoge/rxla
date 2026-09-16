use rxla_core::Tensor;
use rxla_nn::{Cx, Model, ModelInput, Result};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.scope("hidden")?.linear(256).apply(&input)?.relu()?;
    cx.scope("head")?.linear(10).apply(&hidden)
}

fn main() -> Result<()> {
    let model = Model::new(apply).inputs(ModelInput::new([32, 784]));
    let applied = model.trace()?;
    let schema = applied.schema();

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", applied.prepare()?.input_count());
    Ok(())
}
