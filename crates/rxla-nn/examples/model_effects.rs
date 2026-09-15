use rxla_core::Tensor;
use rxla_nn::{Cx, Model, Result};

fn classifier(cx: &mut Cx) -> Result<Tensor> {
    let input = cx.input(&[32, 784])?;
    let hidden = cx.named("hidden")?.linear(256).apply(&input)?.relu()?;
    cx.named("head")?.linear(10).apply(&hidden)
}

fn main() -> Result<()> {
    let model = Model::new(classifier);
    let (schema, applied) = model.trace()?;

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", applied.prepare()?.input_count());
    Ok(())
}
