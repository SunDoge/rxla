use rxla::{
    Tensor,
    nn::{Cx, Linear, Model, ModelInput, Result, TensorApply},
    path,
};

fn apply(cx: Cx, input: Tensor) -> Result<Tensor> {
    let hidden = path!(cx / "hidden")?.layer(Linear::new(256));
    let head = path!(cx / "head")?.layer(Linear::new(10));
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
