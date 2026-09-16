use rxla::{
    Tensor,
    nn::{Cx, Model, ModelInput, Result},
};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.scope("hidden")?.linear(256).apply(&input)?.relu()?;
    cx.scope("head")?.linear(10).apply(&hidden)
}

fn main() -> Result<()> {
    let model = Model::new(apply)
        .inputs(ModelInput::new([32, 784]))
        .trace()?;
    let schema = model.schema();

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", model.prepare()?.input_count());
    Ok(())
}
