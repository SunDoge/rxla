use rxla_core::Tensor;
use rxla_nn::{Cx, Linear, Model, ModelInput, Result, TensorApply};

fn apply(cx: Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.layer("hidden", Linear::new(256))?;
    let head = cx.layer("head", Linear::new(10))?;
    input.apply(&hidden)?.relu()?.apply(&head)
}

fn main() -> Result<()> {
    let model = Model::new(apply).inputs(ModelInput::new([32, 784]));
    let applied = model.trace()?;
    let schema = applied.schema();

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", applied.prepare()?.input_count());
    Ok(())
}
