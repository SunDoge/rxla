//! LoRA parameter discovery for parameter-effect models.

use rxla_nn::{ModelSchema, ParameterSelection};

/// Select standard LoRA adapter matrices while leaving base weights and biases
/// frozen. The recognized canonical suffixes are `lora_A.weight` and
/// `lora_B.weight` at any model-tree depth.
pub fn select_lora(schema: &ModelSchema) -> ParameterSelection {
    schema.select_all().matching(|_, parameter| {
        matches!(
            parameter.path().rsplit_once('.'),
            Some((prefix, "weight"))
                if prefix.ends_with("lora_A") || prefix.ends_with("lora_B")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::Tensor;
    use rxla_nn::{Cx, LoraLinear, Model, Result, TensorApply};

    fn adapter(cx: Cx) -> Result<Tensor> {
        let input = cx.input([2, 4])?;
        let block = cx.scope("block")?;
        input.apply(&block.named_layer("proj", LoraLinear::new(3, 2).bias(false))?)
    }

    #[test]
    fn selection_only_contains_adapter_matrices() {
        let model = Model::new(adapter).trace().unwrap();
        let paths = select_lora(model.schema())
            .parameters()
            .map(|(_, parameter)| parameter.path().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            ["block.proj.lora_A.weight", "block.proj.lora_B.weight"]
        );
    }
}
