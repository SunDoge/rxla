use rxla::{
    Tensor,
    nn::{Cx, Linear, Model, ModelInput, Result},
};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = Linear::new(256).apply(cx, "hidden", &input)?.relu()?;
    Linear::new(10).apply(cx, "head", &hidden)
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
