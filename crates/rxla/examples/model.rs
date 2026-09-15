use rxla::{
    Tensor,
    nn::{Cx, Model, Result},
};

fn classifier(cx: &mut Cx) -> Result<Tensor> {
    let input = cx.input(&[32, 784])?;
    let hidden = cx.named("hidden")?.linear(256).apply(&input)?.relu()?;
    cx.named("head")?.linear(10).apply(&hidden)
}

fn main() -> Result<()> {
    let (schema, model) = Model::new(classifier).trace()?;

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", model.prepare()?.input_count());
    Ok(())
}
