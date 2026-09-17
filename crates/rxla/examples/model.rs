use rxla::{
    Tensor,
    nn::{Cx, Linear, Model, ModelInput, Result},
};

fn apply(cx: &mut Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.apply("hidden", Linear::new(256), &input)?.relu()?;
    cx.apply("head", Linear::new(10), &hidden)
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
