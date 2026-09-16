use rxla::{
    Tensor,
    nn::{Cx, Model, ModelInput, Result},
};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.linear("hidden", &input, 256)?.relu()?;
    cx.linear("head", &hidden, 10)
}

fn main() -> Result<()> {
    let (schema, model) = Model::new(apply)
        .inputs(ModelInput::new([32, 784]))
        .trace()?;

    println!("parameters: {}", schema.parameters().len());
    println!("StableHLO inputs: {}", model.prepare()?.input_count());
    Ok(())
}
