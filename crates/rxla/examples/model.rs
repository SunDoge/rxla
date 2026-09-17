use rxla::{
    Tensor,
    nn::{Cx, Linear, Model, ModelInput, Result, TensorApply},
};

fn apply(cx: Cx, input: Tensor) -> Result<Tensor> {
    let hidden = cx.layer("hidden", Linear::new(256))?;
    let head = cx.layer("head", Linear::new(10))?;
    input.apply(&hidden)?.relu()?.apply(&head)
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
