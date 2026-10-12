//! Trace and verify a model with explicit layer attributes, without PJRT.

use rxla::nn::{Conv2d, GroupNorm, Linear, Model, ModelInput, Result, TensorApply};
use rxla::{Conv2dOptions, Tensor, model};

#[model]
fn image_head(x: Tensor) -> Result<Tensor> {
    #[layer("stem", "conv")]
    let conv = Conv2d::new(16, [3, 3])
        .options(Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            ..Default::default()
        })
        .bias(false);

    #[layer("stem", "norm")]
    let norm = GroupNorm::new(4);

    let x = x.apply(&conv)?.apply(&norm)?.relu()?;
    let x = x.mean([1, 2], false)?;

    #[layer("head")]
    let head = Linear::new(10);
    x.apply(&head)
}

fn main() -> Result<()> {
    let model = Model::new(image_head)
        .inputs(ModelInput::new([2, 32, 32, 3]))
        .trace()?;
    for parameter in model.schema().parameters() {
        println!("{}: {:?}", parameter.path(), parameter.shape());
    }
    model.prepare()?;
    Ok(())
}
