use rxla::{
    Tensor,
    nn::{Cx, Model, ModelInput, Result},
};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.layer("hidden")?.linear(256).apply(&input)?.relu()?;
    cx.layer("head")?.linear(10).apply(&hidden)
}

fn main() -> Result<()> {
    let (schema, model) = Model::new(apply)
        .inputs(ModelInput::new([32, 784]))
        .trace()?;

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", model.prepare()?.input_count());
    Ok(())
}
