//! Functional CLIP text tower used by Stable Diffusion.

use super::*;
use crate::DType;

#[derive(Clone, Debug)]
pub struct ClipTextConfig {
    vocabulary: i64,
    max_positions: i64,
    width: i64,
    intermediate_width: i64,
    layers: usize,
    heads: i64,
    epsilon: f32,
}

impl ClipTextConfig {
    pub fn tiny() -> Self {
        Self {
            vocabulary: 1_000,
            max_positions: 77,
            width: 32,
            intermediate_width: 37,
            layers: 5,
            heads: 4,
            epsilon: 1e-5,
        }
    }

    pub fn stable_diffusion_v1() -> Self {
        Self {
            vocabulary: 49_408,
            max_positions: 77,
            width: 768,
            intermediate_width: 3_072,
            layers: 12,
            heads: 12,
            epsilon: 1e-5,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.vocabulary <= 0
            || self.max_positions <= 0
            || self.width <= 0
            || self.intermediate_width <= 0
            || self.layers == 0
            || self.heads <= 0
            || self.width % self.heads != 0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(Error::InvalidDefinition {
                message: "invalid CLIP text configuration".into(),
            });
        }
        Ok(())
    }
}

fn attention(cx: &mut Cx, input: &Tensor, causal_bias: &Tensor, heads: i64) -> Result<Tensor> {
    let [batch, length, width] = input.shape() else {
        return Err(Error::InvalidDefinition {
            message: "CLIP attention expects [batch, sequence, width]".into(),
        });
    };
    if heads <= 0 || width % heads != 0 {
        return Err(Error::InvalidDefinition {
            message: "CLIP attention heads must divide width".into(),
        });
    }
    let head_dim = width / heads;
    let project = |cx: &mut Cx, name: &str| -> Result<Tensor> {
        Ok(cx
            .layer(name)?
            .linear(*width)
            .apply(input)?
            .reshape(&[*batch, *length, heads, head_dim])?)
    };
    let q = project(cx, "q_proj")?.transpose(&[0, 2, 1, 3])?;
    let k = project(cx, "k_proj")?.transpose(&[0, 2, 1, 3])?;
    let v = project(cx, "v_proj")?.transpose(&[0, 2, 1, 3])?;
    let hidden = q
        .scaled_dot_product_attention(&k, &v, Some(causal_bias), None)?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[*batch, *length, *width])?;
    cx.layer("out_proj")?.linear(*width).apply(&hidden)
}

fn quick_gelu(input: &Tensor) -> Result<Tensor> {
    Ok(input.mul(&input.mul_scalar(1.702)?.sigmoid()?)?)
}

fn encoder_layer(
    cx: &mut Cx,
    input: &Tensor,
    causal_bias: &Tensor,
    config: &ClipTextConfig,
) -> Result<Tensor> {
    let normalized = cx
        .layer("layer_norm1")?
        .layer_norm(1)
        .epsilon(config.epsilon)
        .apply(input)?;
    let attended = {
        let mut scope = cx.scope("self_attn")?;
        attention(&mut scope, &normalized, causal_bias, config.heads)?
    };
    let hidden = input.add(&attended)?;
    let normalized = cx
        .layer("layer_norm2")?
        .layer_norm(1)
        .epsilon(config.epsilon)
        .apply(&hidden)?;
    let feed_forward = {
        let mut scope = cx.scope("mlp")?;
        let projected = scope
            .layer("fc1")?
            .linear(config.intermediate_width)
            .apply(&normalized)?;
        scope
            .layer("fc2")?
            .linear(config.width)
            .apply(&quick_gelu(&projected)?)
    }?;
    Ok(hidden.add(&feed_forward)?)
}

/// Encode I32 token IDs `[batch, sequence]` into CLIP hidden states.
pub fn clip_text_encoder(
    cx: &mut Cx,
    token_ids: &Tensor,
    config: &ClipTextConfig,
) -> Result<Tensor> {
    config.validate()?;
    let [_, length] = token_ids.shape() else {
        return Err(Error::InvalidDefinition {
            message: "CLIP token IDs must have shape [batch, sequence]".into(),
        });
    };
    if token_ids.dtype() != DType::I32 || *length <= 0 || *length > config.max_positions {
        return Err(Error::InvalidDefinition {
            message: "CLIP token IDs or sequence length are invalid".into(),
        });
    }
    let positions = cx.iota_i32(token_ids.shape(), 1)?;
    let mut hidden = {
        let mut scope = cx.scope_path(["text_model", "embeddings"])?;
        let tokens = scope
            .layer("token_embedding")?
            .embedding(config.vocabulary, config.width)
            .apply(token_ids)?;
        let positions = scope
            .layer("position_embedding")?
            .embedding(config.max_positions, config.width)
            .apply(&positions)?;
        tokens.add(&positions)?
    };
    let bias_shape = [1, 1, *length, *length];
    let query = cx.iota_i32(&bias_shape, 2)?.to_f32()?;
    let key = cx.iota_i32(&bias_shape, 3)?.to_f32()?;
    let allowed = query.ge_mask(&key)?;
    let zero = cx.constant(&[], &[0.0])?.broadcast_to(&bias_shape)?;
    let blocked = cx
        .constant(&[], &[f32::NEG_INFINITY])?
        .broadcast_to(&bias_shape)?;
    let causal_bias = allowed.select(&zero, &blocked)?;
    {
        let mut layers = cx.scope_path(["text_model", "encoder", "layers"])?;
        for layer in 0..config.layers {
            let mut scope = layers.scope(&layer.to_string())?;
            hidden = encoder_layer(&mut scope, &hidden, &causal_bias, config)?;
        }
    }
    cx.scope("text_model")?
        .layer("final_layer_norm")?
        .layer_norm(1)
        .epsilon(config.epsilon)
        .apply(&hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::{apply, init};

    fn tiny(cx: &mut Cx) -> Result<Tensor> {
        let ids = cx.input_dtype(&[2, 77], DType::I32)?;
        clip_text_encoder(cx, &ids, &ClipTextConfig::tiny())
    }

    #[test]
    fn tiny_clip_has_stable_schema_and_lowers() {
        let (schema, output) = init(tiny).unwrap();
        assert_eq!(output.shape(), [2, 77, 32]);
        assert_eq!(schema.parameters().len(), 84);
        assert_eq!(
            schema
                .get("text_model.embeddings.token_embedding.weight")
                .unwrap()
                .shape(),
            [1000, 32]
        );
        assert!(
            schema
                .get("text_model.encoder.layers.1.mlp.fc2.bias")
                .is_some()
        );
        apply(&schema, tiny).unwrap().prepare().unwrap();
    }
}
