use rxla_core::Tensor;
use rxla_nn::{Cx, Layer, Linear, Model, ModelInput, Result};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = Linear::new(256).named("hidden");
    let head = Linear::new(10).named("head");
    let input = hidden.apply(cx, &input)?.relu()?;
    head.apply(cx, &input)
}

fn main() -> Result<()> {
    let model = Model::new(apply).inputs(ModelInput::new([32, 784]));
    let applied = model.trace()?;
    let schema = applied.schema();

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", applied.prepare()?.input_count());
    Ok(())
}
