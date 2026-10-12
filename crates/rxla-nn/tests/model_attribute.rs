use rxla_core::Tensor;
use rxla_nn::{Linear, Model, ModelInput, Result, TensorApply, model};

#[model]
fn classifier(x: Tensor) -> Result<Tensor> {
    #[layer("head")]
    let head = Linear::new(3);
    x.apply(&head)
}

#[model]
fn repeated(x: Tensor) -> Result<Tensor> {
    let mut x = x;
    for index in 0usize..2 {
        #[layer("blocks", index, "linear")]
        let projection = Linear::new(4).bias(false);
        x = x.apply(&projection)?;
    }
    // Same explicit path means the same parameter, not another declaration.
    #[layer("blocks", 0usize, "linear")]
    let shared = Linear::new(4).bias(false);
    x.apply(&shared)
}

#[model]
fn unused_layer(x: Tensor) -> Result<Tensor> {
    #[layer("unused")]
    let _projection = Linear::new(100);
    Ok(x)
}

#[model]
fn invalid_dynamic_path(x: Tensor) -> Result<Tensor> {
    let name = String::from("invalid.name");
    #[layer(name)]
    let projection = Linear::new(3);
    x.apply(&projection)
}

#[model]
fn hygiene(x: Tensor) -> Result<Tensor> {
    let __rxla_model_cx = 42;
    #[layer("head")]
    let head = Linear::new(__rxla_model_cx);
    x.apply(&head)
}

#[model]
fn evaluates_once(x: Tensor) -> Result<Tensor> {
    let mut path_calls = 0;
    let mut layer_calls = 0;
    #[layer({ path_calls += 1; "head" })]
    let head = {
        layer_calls += 1;
        Linear::new(3)
    };
    assert_eq!((path_calls, layer_calls), (1, 1));
    x.apply(&head)
}

#[test]
fn model_function_traces_without_user_context_or_runtime() -> Result<()> {
    let applied = Model::new(classifier)
        .inputs(ModelInput::new([2, 4]))
        .trace()?;
    let paths = applied
        .schema()
        .parameters()
        .iter()
        .map(|p| (p.path(), p.shape()))
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        [("head.weight", &[3, 4][..]), ("head.bias", &[3][..])]
    );
    applied.prepare()?;
    Ok(())
}

#[test]
fn loop_paths_and_shared_parameters_keep_explicit_identity() -> Result<()> {
    let applied = Model::new(repeated)
        .inputs(ModelInput::new([2, 4]))
        .trace()?;
    let paths = applied
        .schema()
        .parameters()
        .iter()
        .map(|p| p.path())
        .collect::<Vec<_>>();
    assert_eq!(paths, ["blocks.0.linear.weight", "blocks.1.linear.weight"]);
    applied.prepare()?;
    Ok(())
}

#[test]
fn binding_does_not_create_unused_parameters() -> Result<()> {
    let applied = Model::new(unused_layer)
        .inputs(ModelInput::new([2, 4]))
        .trace()?;
    assert!(applied.schema().parameters().is_empty());
    Ok(())
}

#[test]
fn dynamic_paths_remain_fallible_and_context_is_hygienic() -> Result<()> {
    assert!(
        Model::new(invalid_dynamic_path)
            .inputs(ModelInput::new([2, 4]))
            .trace()
            .is_err()
    );
    let applied = Model::new(hygiene)
        .inputs(ModelInput::new([2, 4]))
        .trace()?;
    assert_eq!(applied.schema().parameters()[0].shape(), [42, 4]);
    Model::new(evaluates_once)
        .inputs(ModelInput::new([2, 4]))
        .trace()?;
    Ok(())
}
