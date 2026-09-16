//! Functional Stable Diffusion building blocks using scoped parameter effects.

use rxla_core::{Conv2dOptions, DType, Tensor};
use rxla_nn::{Cx, Scope};
use snafu::Snafu;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Nn { source: rxla_nn::Error },
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
    #[snafu(display("invalid model definition: {reason}"))]
    InvalidModel { reason: &'static str },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

mod unet;
pub use unet::{UnetConfig, unet};
mod clip;
pub use clip::{ClipTextConfig, clip_text_encoder};
mod vae;
pub use vae::{AutoencoderKlDecoderConfig, autoencoder_kl_decoder};
mod scheduler;
pub use scheduler::{PndmSampleSource, PndmScheduler, PndmStep};
mod taesd;
pub use taesd::taesd_decoder;
mod qwen3_5;
pub use qwen3_5::{Qwen3_5Config, qwen3_5};
pub mod pp_ocr_v6;

/// Learned two-layer projection for a sinusoidal timestep embedding.
/// The input width is inferred at the point of use.
pub fn timestep_embedding(cx: &mut Cx, input: &Tensor, output_width: i64) -> Result<Tensor> {
    let hidden = cx.scope("linear_1")?.linear(output_width).apply(input)?;
    Ok(cx
        .scope("linear_2")?
        .linear(output_width)
        .apply(&hidden.silu()?)?)
}

/// Static choices for a Diffusers-compatible `ResnetBlock2D`.
#[derive(Clone, Copy, Debug)]
pub struct Resnet2dOptions {
    out_channels: i64,
    groups: i64,
    epsilon: f32,
}

impl Resnet2dOptions {
    pub fn new(out_channels: i64) -> Self {
        Self {
            out_channels,
            groups: 32,
            epsilon: 1e-5,
        }
    }

    pub fn groups(mut self, groups: i64) -> Self {
        self.groups = groups;
        self
    }

    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Diffusers-compatible additive-timestep ResNet block.
///
/// The caller supplies the block's lexical scope. Nested paths are exactly
/// `norm1`, `conv1`, `time_emb_proj`, `norm2`, `conv2`, and, when channels
/// change, `conv_shortcut`.
pub fn resnet2d(
    cx: &mut Cx,
    input: &Tensor,
    timestep_embedding: &Tensor,
    options: Resnet2dOptions,
) -> Result<Tensor> {
    if input.shape().len() != 4 || timestep_embedding.shape().len() != 2 {
        return Err(Error::InvalidModel {
            reason: "stable diffusion ResNet expects NHWC input and rank-two timestep embedding",
        });
    }
    if input.shape()[0] != timestep_embedding.shape()[0] {
        return Err(Error::InvalidModel {
            reason: "stable diffusion ResNet input and timestep batches must match",
        });
    }
    let convolution = Conv2dOptions {
        padding: [[1, 1], [1, 1]],
        ..Default::default()
    };
    let normalized = cx
        .scope("norm1")?
        .group_norm(options.groups)
        .epsilon(options.epsilon)
        .apply(input)?;
    let mut hidden = cx
        .scope("conv1")?
        .conv2d(options.out_channels, [3, 3])
        .options(convolution)
        .apply(&normalized.silu()?)?;
    let time = cx
        .scope("time_emb_proj")?
        .linear(options.out_channels)
        .apply(&timestep_embedding.silu()?)?
        .reshape(&[timestep_embedding.shape()[0], 1, 1, options.out_channels])?
        .broadcast_to(hidden.shape())?;
    hidden = hidden.add(&time)?;
    let normalized = cx
        .scope("norm2")?
        .group_norm(options.groups)
        .epsilon(options.epsilon)
        .apply(&hidden)?;
    hidden = cx
        .scope("conv2")?
        .conv2d(options.out_channels, [3, 3])
        .options(convolution)
        .apply(&normalized.silu()?)?;
    let residual = if input.shape()[3] == options.out_channels {
        input.clone()
    } else {
        cx.scope("conv_shortcut")?
            .conv2d(options.out_channels, [1, 1])
            .apply(input)?
    };
    Ok(hidden.add(&residual)?)
}

/// Static choices for a Diffusers-compatible spatial transformer.
#[derive(Clone, Copy, Debug)]
pub struct SpatialTransformerOptions {
    head_dim: i64,
    layers: usize,
    groups: i64,
}

impl SpatialTransformerOptions {
    pub fn new(head_dim: i64) -> Self {
        Self {
            head_dim,
            layers: 1,
            groups: 32,
        }
    }

    pub fn layers(mut self, layers: usize) -> Self {
        self.layers = layers;
        self
    }

    pub fn groups(mut self, groups: i64) -> Self {
        self.groups = groups;
        self
    }
}

fn cross_attention(cx: &mut Cx, query: &Tensor, context: &Tensor, head_dim: i64) -> Result<Tensor> {
    if query.shape().len() != 3
        || context.shape().len() != 3
        || query.shape()[0] != context.shape()[0]
    {
        return Err(Error::InvalidModel {
            reason: "cross attention expects rank-three query/context with equal batches",
        });
    }
    let [batch, query_length, width] = query.shape() else {
        unreachable!("rank checked above")
    };
    if head_dim <= 0 || *width <= 0 || width % head_dim != 0 {
        return Err(Error::InvalidModel {
            reason: "attention width must be divisible by positive head_dim",
        });
    }
    let heads = width / head_dim;
    let context_length = context.shape()[1];
    let q = cx
        .scope("to_q")?
        .linear(*width)
        .bias(false)
        .apply(query)?
        .reshape(&[*batch, *query_length, heads, head_dim])?
        .transpose(&[0, 2, 1, 3])?;
    let k = cx
        .scope("to_k")?
        .linear(*width)
        .bias(false)
        .apply(context)?
        .reshape(&[*batch, context_length, heads, head_dim])?
        .transpose(&[0, 2, 1, 3])?;
    let v = cx
        .scope("to_v")?
        .linear(*width)
        .bias(false)
        .apply(context)?
        .reshape(&[*batch, context_length, heads, head_dim])?
        .transpose(&[0, 2, 1, 3])?;
    let hidden = q
        .scaled_dot_product_attention(&k, &v, None, None)?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[*batch, *query_length, *width])?;
    Ok(cx
        .scope("to_out")?
        .scope("0")?
        .linear(*width)
        .apply(&hidden)?)
}

fn feed_forward(cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
    let width = *input.shape().last().ok_or(Error::InvalidModel {
        reason: "feed-forward input must have a feature dimension",
    })?;
    let hidden = width.checked_mul(4).ok_or(Error::InvalidModel {
        reason: "GEGLU hidden width overflow",
    })?;
    let projected = hidden.checked_mul(2).ok_or(Error::InvalidModel {
        reason: "GEGLU projected width overflow",
    })?;
    let projected = cx
        .scope("net")?
        .scope("0")?
        .scope("proj")?
        .linear(projected)
        .apply(input)?;
    let parts = projected.split(projected.shape().len() - 1, &[hidden, hidden])?;
    let gated = parts[0].mul(&parts[1].gelu()?)?;
    Ok(cx.scope("net")?.scope("2")?.linear(width).apply(&gated)?)
}

fn transformer_block(
    cx: &mut Cx,
    input: &Tensor,
    context: &Tensor,
    head_dim: i64,
) -> Result<Tensor> {
    let normalized = cx.scope("norm1")?.layer_norm(1).apply(input)?;
    let attention = {
        let mut scope = cx.scope("attn1")?;
        cross_attention(&mut scope, &normalized, &normalized, head_dim)?
    };
    let hidden = input.add(&attention)?;
    let normalized = cx.scope("norm2")?.layer_norm(1).apply(&hidden)?;
    let attention = {
        let mut scope = cx.scope("attn2")?;
        cross_attention(&mut scope, &normalized, context, head_dim)?
    };
    let hidden = hidden.add(&attention)?;
    let normalized = cx.scope("norm3")?.layer_norm(1).apply(&hidden)?;
    let mut ff = cx.scope("ff")?;
    Ok(hidden.add(&feed_forward(&mut ff, &normalized)?)?)
}

/// Diffusers `Transformer2DModel` for NHWC activations and `[B,T,C]` context.
pub fn spatial_transformer(
    cx: &mut Cx,
    input: &Tensor,
    context: &Tensor,
    options: SpatialTransformerOptions,
) -> Result<Tensor> {
    if input.shape().len() != 4
        || context.shape().len() != 3
        || input.shape()[0] != context.shape()[0]
        || options.layers == 0
    {
        return Err(Error::InvalidModel {
            reason: "spatial transformer expects NHWC input, rank-three context, equal batches and layers",
        });
    }
    let [batch, height, width, channels] = input.shape() else {
        unreachable!("rank checked above")
    };
    let normalized = cx
        .scope("norm")?
        .group_norm(options.groups)
        .epsilon(1e-6)
        .apply(input)?;
    let mut hidden = cx
        .scope("proj_in")?
        .conv2d(*channels, [1, 1])
        .apply(&normalized)?
        .reshape(&[*batch, height * width, *channels])?;
    for layer in 0..options.layers {
        let mut scope = cx.scope_path(["transformer_blocks".to_owned(), layer.to_string()])?;
        hidden = transformer_block(&mut scope, &hidden, context, options.head_dim)?;
    }
    let hidden = hidden.reshape(&[*batch, *height, *width, *channels])?;
    Ok(cx
        .scope("proj_out")?
        .conv2d(*channels, [1, 1])
        .apply(&hidden)?
        .add(input)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::Model;

    fn model(cx: &mut Cx) -> Result<Tensor> {
        let image = cx.input(&[2, 16, 12, 32])?;
        let timestep = cx.input(&[2, 128])?;
        let mut scope = cx.scope("block")?;
        resnet2d(&mut scope, &image, &timestep, Resnet2dOptions::new(64))
    }

    #[test]
    fn timestep_mlp_infers_input_width() {
        let model = Model::new(|cx: &mut Cx| {
            let input = cx.input(&[2, 32])?;
            let mut scope = cx.scope("time_embedding")?;
            timestep_embedding(&mut scope, &input, 128)
        })
        .trace()
        .unwrap();
        let schema = model.schema();
        assert_eq!(model.outputs()[0].shape(), [2, 128]);
        assert_eq!(
            schema
                .get("time_embedding.linear_1.weight")
                .unwrap()
                .shape(),
            [128, 32]
        );
        assert_eq!(
            schema
                .get("time_embedding.linear_2.weight")
                .unwrap()
                .shape(),
            [128, 128]
        );
    }

    #[test]
    fn resnet_infers_shapes_and_diffusers_paths() {
        let model = Model::new(model).trace().unwrap();
        let schema = model.schema();
        assert_eq!(model.outputs()[0].shape(), [2, 16, 12, 64]);
        assert_eq!(schema.parameters().len(), 12);
        for path in [
            "block.norm1.weight",
            "block.conv1.weight",
            "block.time_emb_proj.weight",
            "block.norm2.bias",
            "block.conv2.weight",
            "block.conv_shortcut.bias",
        ] {
            assert!(schema.get(path).is_some(), "missing {path}");
        }
        assert_eq!(model.prepare().unwrap().input_count(), 14);
    }

    #[test]
    fn equal_width_resnet_omits_shortcut_parameters() {
        let model = Model::new(|cx: &mut Cx| {
            let image = cx.input(&[1, 8, 8, 32])?;
            let timestep = cx.input(&[1, 128])?;
            resnet2d(cx, &image, &timestep, Resnet2dOptions::new(32))
        })
        .trace()
        .unwrap();
        let schema = model.schema();
        assert_eq!(model.outputs()[0].shape(), [1, 8, 8, 32]);
        assert_eq!(schema.parameters().len(), 10);
        assert!(schema.get("conv_shortcut.weight").is_none());
    }

    #[test]
    fn spatial_transformer_infers_width_and_diffusers_paths() {
        let model = |cx: &mut Cx| {
            let image = cx.input(&[2, 8, 6, 64])?;
            let context = cx.input(&[2, 77, 32])?;
            spatial_transformer(cx, &image, &context, SpatialTransformerOptions::new(8))
        };
        let model = Model::new(model).trace().unwrap();
        let schema = model.schema();
        assert_eq!(model.outputs()[0].shape(), [2, 8, 6, 64]);
        assert_eq!(schema.parameters().len(), 26);
        for path in [
            "norm.weight",
            "proj_in.weight",
            "transformer_blocks.0.attn1.to_q.weight",
            "transformer_blocks.0.attn2.to_k.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
            "transformer_blocks.0.ff.net.2.bias",
            "proj_out.bias",
        ] {
            assert!(schema.get(path).is_some(), "missing {path}");
        }
        model.prepare().unwrap();
    }
}
