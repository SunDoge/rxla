use rxla::{
    Tensor,
    nn::{Cx, Model, Result},
};

fn apply(cx: &mut Cx) -> Result<Tensor> {
    let input = cx.input(&[32, 784])?;
    let hidden = cx.named("hidden")?.linear(256).apply(&input)?.relu()?;
    cx.named("head")?.linear(10).apply(&hidden)
}

fn main() -> Result<()> {
    let (schema, model) = Model::new(apply).trace()?;

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", model.prepare()?.input_count());
    Ok(())
}
