use rxla::nn::{Linear, Model, ModelInput, Result, TensorApply};
use rxla::{Tensor, model};

#[model]
fn classifier(x: Tensor) -> Result<Tensor> {
    #[layer("head")]
    let head = Linear::new(3);
    x.apply(&head)
}

#[test]
fn facade_macro_needs_no_direct_nn_dependency() -> Result<()> {
    let applied = Model::new(classifier)
        .inputs(ModelInput::new([2, 4]))
        .trace()?;
    assert_eq!(applied.schema().parameters()[0].path(), "head.weight");
    applied.prepare()?;
    Ok(())
}
