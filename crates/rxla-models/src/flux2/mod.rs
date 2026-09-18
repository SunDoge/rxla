//! FLUX.2 Klein rectified-flow transformer.
//!
//! Parameter paths intentionally match the official Black Forest Labs
//! checkpoint so safetensors can bind without a model-specific rename table.

mod blocks;
mod embedding;
mod scheduler;

pub use scheduler::{Flux2EulerScheduler, Flux2EulerStep};

use self::blocks::{DoubleStreamModulation, Modulation, double_block, single_block};
use self::embedding::{multi_axis_rope, timestep_embedding};
use super::{Error, ModelDefinitionError, Result};
use rxla_core::{DType, Tensor};
use rxla_nn::{Cx, Linear, TensorApply};

/// Static topology of a FLUX.2 transformer.
#[derive(Clone, Debug)]
pub struct Flux2Config {
    in_channels: i64,
    context_in_dim: i64,
    hidden_size: i64,
    num_heads: i64,
    depth: usize,
    depth_single_blocks: usize,
    axes_dims_rope: Vec<i64>,
    rope_theta: f32,
    mlp_ratio: i64,
    timestep_channels: i64,
    epsilon: f32,
    parameter_dtype: DType,
}

impl Flux2Config {
    /// Official Apache-2.0 FLUX.2 Klein 4B topology.
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            context_in_dim: 7680,
            hidden_size: 3072,
            num_heads: 24,
            depth: 5,
            depth_single_blocks: 20,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000.0,
            mlp_ratio: 3,
            timestep_channels: 256,
            epsilon: 1e-6,
            parameter_dtype: DType::BF16,
        }
    }

    /// Small shape-compatible topology for compiler and numerical conformance
    /// tests. It preserves all FLUX.2 block kinds and four-axis RoPE.
    pub fn tiny() -> Self {
        Self {
            in_channels: 8,
            context_in_dim: 12,
            hidden_size: 32,
            num_heads: 4,
            depth: 1,
            depth_single_blocks: 1,
            axes_dims_rope: vec![2, 2, 2, 2],
            rope_theta: 2000.0,
            mlp_ratio: 2,
            timestep_channels: 256,
            epsilon: 1e-6,
            parameter_dtype: DType::F32,
        }
    }

    fn head_dim(&self) -> i64 {
        self.hidden_size / self.num_heads
    }

    fn validate(&self) -> Result<()> {
        let rope_width: i64 = self.axes_dims_rope.iter().sum();
        let valid = self.in_channels > 0
            && self.context_in_dim > 0
            && self.hidden_size > 0
            && self.num_heads > 0
            && self.hidden_size % self.num_heads == 0
            && self.depth > 0
            && self.depth_single_blocks > 0
            && !self.axes_dims_rope.is_empty()
            && self
                .axes_dims_rope
                .iter()
                .all(|&width| width > 0 && width % 2 == 0)
            && rope_width == self.head_dim()
            && self.rope_theta.is_finite()
            && self.rope_theta > 0.0
            && self.mlp_ratio > 0
            && self.timestep_channels > 0
            && self.timestep_channels % 2 == 0
            && self.epsilon.is_finite()
            && self.epsilon > 0.0
            && matches!(self.parameter_dtype, DType::BF16 | DType::F32);
        if !valid {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidFlux2Configuration,
            });
        }
        Ok(())
    }
}

fn linear(cx: Cx, input: &Tensor, width: i64) -> Result<Tensor> {
    Ok(input.through(cx, Linear::new(width).bias(false))?)
}

/// Apply the FLUX.2 denoising transformer.
///
/// `image` and `context` are token sequences. Their IDs contain the four
/// `(t, h, w, l)` coordinates consumed by FLUX.2's multi-axis RoPE.
pub fn flux2(
    cx: Cx,
    image: &Tensor,
    image_ids: &Tensor,
    timestep: &Tensor,
    context: &Tensor,
    context_ids: &Tensor,
    config: &Flux2Config,
) -> Result<Tensor> {
    config.validate()?;
    let cx = cx.with_parameter_storage_dtype(config.parameter_dtype)?;
    let ([batch, image_tokens, image_width], [context_batch, context_tokens, context_width]) =
        (image.shape(), context.shape())
    else {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidFlux2Input,
        });
    };
    let valid = image.dtype() == DType::F32
        && context.dtype() == DType::F32
        && image_ids.dtype() == DType::F32
        && context_ids.dtype() == DType::F32
        && timestep.dtype() == DType::F32
        && *batch == *context_batch
        && *image_width == config.in_channels
        && *context_width == config.context_in_dim
        && image_ids.shape() == [*batch, *image_tokens, config.axes_dims_rope.len() as i64]
        && context_ids.shape() == [*batch, *context_tokens, config.axes_dims_rope.len() as i64]
        && timestep.shape() == [*batch];
    if !valid {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidFlux2Input,
        });
    }

    let vector = timestep_embedding(
        cx.scope("time_in")?,
        timestep,
        config.timestep_channels,
        config.hidden_size,
    )?;
    let image_modulation = DoubleStreamModulation::new(
        cx.scope("double_stream_modulation_img")?,
        &vector,
        config.hidden_size,
    )?;
    let text_modulation = DoubleStreamModulation::new(
        cx.scope("double_stream_modulation_txt")?,
        &vector,
        config.hidden_size,
    )?;
    let single_modulation = Modulation::new(
        cx.scope("single_stream_modulation")?,
        &vector,
        config.hidden_size,
    )?;

    let mut image = linear(cx.scope("img_in")?, image, config.hidden_size)?;
    let mut context = linear(cx.scope("txt_in")?, context, config.hidden_size)?;
    let image_rope = multi_axis_rope(cx.clone(), image_ids, config)?;
    let context_rope = multi_axis_rope(cx.clone(), context_ids, config)?;

    let blocks = cx.scope("double_blocks")?;
    for index in 0..config.depth {
        (image, context) = double_block(
            blocks.scope(index.to_string())?,
            &image,
            &context,
            &image_rope,
            &context_rope,
            &image_modulation,
            &text_modulation,
            config,
        )?;
    }

    let mut hidden = Tensor::concatenate(&[context, image], 1)?;
    let rope = Tensor::concatenate(&[context_rope, image_rope], 2)?;
    let blocks = cx.scope("single_blocks")?;
    for index in 0..config.depth_single_blocks {
        hidden = single_block(
            blocks.scope(index.to_string())?,
            &hidden,
            &rope,
            &single_modulation,
            config,
        )?;
    }
    let hidden = hidden.narrow(1, *context_tokens, *image_tokens)?;
    let final_layer = cx.scope("final_layer")?;
    let normalized = hidden.layer_norm(&[config.hidden_size], None, None, config.epsilon)?;
    let modulation = linear(
        final_layer.scope("adaLN_modulation")?.scope("1")?,
        &vector.silu()?,
        config.hidden_size * 2,
    )?
    .split(1, &[config.hidden_size, config.hidden_size])?;
    let shift = modulation[0]
        .unsqueeze(1)?
        .broadcast_to(normalized.shape())?;
    let scale = modulation[1]
        .unsqueeze(1)?
        .broadcast_to(normalized.shape())?;
    let normalized = normalized.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;
    linear(
        final_layer.scope("linear")?,
        &normalized,
        config.in_channels,
    )
}

/// One reusable device-resident rectified-flow denoising step.
///
/// `step_size` has shape `[batch]` and contains `next_timestep - timestep`.
/// Keeping it as an input lets one compiled executable serve every Klein step;
/// the transformer prediction and latent update remain in one graph.
#[allow(clippy::too_many_arguments)]
pub fn flux2_euler_step(
    cx: Cx,
    sample: &Tensor,
    image_ids: &Tensor,
    timestep: &Tensor,
    step_size: &Tensor,
    context: &Tensor,
    context_ids: &Tensor,
    config: &Flux2Config,
) -> Result<Tensor> {
    let prediction = flux2(
        cx,
        sample,
        image_ids,
        timestep,
        context,
        context_ids,
        config,
    )?;
    if step_size.dtype() != DType::F32 || step_size.shape() != [sample.shape()[0]] {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidFlux2Input,
        });
    }
    let coefficient = step_size
        .reshape([sample.shape()[0], 1, 1])?
        .broadcast_to(sample.shape())?;
    Ok(sample.add(&prediction.mul(&coefficient)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::Model;

    #[test]
    fn tiny_flux2_traces_with_official_parameter_paths() {
        let config = Flux2Config::tiny();
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 3, 8])?;
            let image_ids = cx.input([1, 3, 4])?;
            let timestep = cx.input([1])?;
            let context = cx.input([1, 2, 12])?;
            let context_ids = cx.input([1, 2, 4])?;
            flux2(
                cx,
                &image,
                &image_ids,
                &timestep,
                &context,
                &context_ids,
                &config,
            )
        })
        .trace()
        .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 3, 8]);
        let schema = model.schema();
        for path in [
            "img_in.weight",
            "time_in.in_layer.weight",
            "double_stream_modulation_img.lin.weight",
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.txt_attn.norm.query_norm.scale",
            "single_blocks.0.linear1.weight",
            "single_blocks.0.norm.key_norm.scale",
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.linear.weight",
        ] {
            assert!(schema.get(path).is_some(), "missing {path}");
        }
        model.prepare().unwrap();
    }

    #[test]
    fn klein_4b_topology_matches_official_dimensions() {
        let config = Flux2Config::klein_4b();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_heads, 24);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.depth, 5);
        assert_eq!(config.depth_single_blocks, 20);
        config.validate().unwrap();
    }

    #[test]
    fn klein_4b_schema_has_the_official_parameter_count() {
        let config = Flux2Config::klein_4b();
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 1, 128])?;
            let image_ids = cx.input([1, 1, 4])?;
            let timestep = cx.input([1])?;
            let context = cx.input([1, 1, 7680])?;
            let context_ids = cx.input([1, 1, 4])?;
            flux2(
                cx,
                &image,
                &image_ids,
                &timestep,
                &context,
                &context_ids,
                &config,
            )
        })
        .trace()
        .unwrap();
        let parameters = model.schema().parameters();
        assert_eq!(parameters.len(), 149);
        assert!(
            parameters
                .iter()
                .all(|parameter| parameter.storage_dtype() == DType::BF16)
        );
        let elements = parameters
            .iter()
            .map(|parameter| parameter.shape().iter().product::<i64>())
            .sum::<i64>();
        assert_eq!(elements, 3_875_544_576);
    }

    #[test]
    fn reusable_euler_step_keeps_one_parameter_tree() {
        let config = Flux2Config::tiny();
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 3, 8])?;
            let image_ids = cx.input([1, 3, 4])?;
            let timestep = cx.input([1])?;
            let step_size = cx.input([1])?;
            let context = cx.input([1, 2, 12])?;
            let context_ids = cx.input([1, 2, 4])?;
            flux2_euler_step(
                cx,
                &image,
                &image_ids,
                &timestep,
                &step_size,
                &context,
                &context_ids,
                &config,
            )
        })
        .trace()
        .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 3, 8]);
        assert_eq!(model.schema().parameters().len(), 25);
        model.prepare().unwrap();
    }
}
