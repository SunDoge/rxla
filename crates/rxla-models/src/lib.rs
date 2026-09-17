//! Functional Stable Diffusion building blocks using scoped parameter effects.

use rxla_core::{Conv2dOptions, DType, Tensor};
use rxla_nn::{Cx, LayerNorm, Linear, TensorApply};
use snafu::Snafu;

/// Machine-readable model definition and input validation failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Snafu)]
#[non_exhaustive]
pub enum ModelDefinitionError {
    #[snafu(display("invalid Stable Diffusion ResNet input"))]
    InvalidStableDiffusionResnetInput,
    #[snafu(display("Stable Diffusion ResNet input and timestep batches differ"))]
    StableDiffusionResnetBatchMismatch,
    #[snafu(display("invalid cross-attention input"))]
    InvalidCrossAttentionInput,
    #[snafu(display("attention width is not divisible by a positive head dimension"))]
    InvalidAttentionHeadDimension,
    #[snafu(display("feed-forward input has no feature dimension"))]
    MissingFeedForwardDimension,
    #[snafu(display("GEGLU hidden width overflow"))]
    GegluHiddenWidthOverflow,
    #[snafu(display("GEGLU projected width overflow"))]
    GegluProjectedWidthOverflow,
    #[snafu(display("invalid spatial-transformer input"))]
    InvalidSpatialTransformerInput,
    #[snafu(display("invalid Stable Diffusion UNet configuration"))]
    InvalidUnetConfiguration,
    #[snafu(display("UNet input shapes do not match its configuration"))]
    InvalidUnetInput,
    #[snafu(display("UNet timestep width overflow"))]
    UnetTimestepWidthOverflow,
    #[snafu(display("UNet has too few down-block residuals"))]
    MissingUnetResidual,
    #[snafu(display("UNet forward left unused residuals"))]
    UnusedUnetResidual,
    #[snafu(display("invalid CLIP text configuration"))]
    InvalidClipConfiguration,
    #[snafu(display("CLIP attention expects rank-three input"))]
    InvalidClipAttentionInput,
    #[snafu(display("CLIP attention heads do not divide its width"))]
    InvalidClipAttentionHeads,
    #[snafu(display("CLIP token IDs must have shape [batch, sequence]"))]
    InvalidClipTokenShape,
    #[snafu(display("CLIP token IDs or sequence length are invalid"))]
    InvalidClipTokens,
    #[snafu(display("invalid AutoencoderKL decoder configuration"))]
    InvalidVaeConfiguration,
    #[snafu(display("VAE attention expects NHWC input"))]
    InvalidVaeAttentionInput,
    #[snafu(display("VAE decoder latent shape does not match its configuration"))]
    InvalidVaeLatentShape,
    #[snafu(display("VAE configuration has no block channels"))]
    MissingVaeBlockChannels,
    #[snafu(display("TAESD decoder expects NHWC input with four latent channels"))]
    InvalidTaesdLatentShape,
    #[snafu(display("guidance scale must be finite"))]
    NonFiniteGuidanceScale,
    #[snafu(display("PNDM inference steps must be in 2..=1000"))]
    InvalidPndmInferenceSteps,
    #[snafu(display("PNDM timestep ratio must be positive"))]
    InvalidPndmTimestepRatio,
    #[snafu(display("PNDM timestep exceeds i64"))]
    PndmTimestepTooLarge,
    #[snafu(display("PNDM timestep ratio exceeds i64"))]
    PndmTimestepRatioTooLarge,
    #[snafu(display("PNDM step index is out of range"))]
    PndmStepOutOfRange,
    #[snafu(display("PNDM timestep overflow"))]
    PndmTimestepOverflow,
    #[snafu(display("PNDM produced nonfinite update coefficients"))]
    NonFinitePndmCoefficients,
    #[snafu(display("PNDM timestep is outside the training schedule"))]
    PndmTimestepOutsideSchedule,
    #[snafu(display("invalid Qwen3.5 configuration"))]
    InvalidQwenConfiguration,
    #[snafu(display("Qwen3.5 token IDs must have shape [batch, sequence]"))]
    InvalidQwenTokenShape,
    #[snafu(display("Qwen3.5 requires nonempty in-range I32 token IDs"))]
    InvalidQwenTokens,
}

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Nn { source: rxla_nn::Error },
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
    #[snafu(display("invalid model definition: {kind}"))]
    InvalidModel { kind: ModelDefinitionError },
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
pub fn timestep_embedding(cx: Cx, input: &Tensor, output_width: i64) -> Result<Tensor> {
    let hidden = input.apply(&cx.layer("linear_1", Linear::new(output_width))?)?;
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
    cx: Cx,
    input: &Tensor,
    timestep_embedding: &Tensor,
    options: Resnet2dOptions,
) -> Result<Tensor> {
    if input.shape().len() != 4 || timestep_embedding.shape().len() != 2 {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidStableDiffusionResnetInput,
        });
    }
    if input.shape()[0] != timestep_embedding.shape()[0] {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::StableDiffusionResnetBatchMismatch,
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

fn cross_attention(cx: Cx, query: &Tensor, context: &Tensor, head_dim: i64) -> Result<Tensor> {
    if query.shape().len() != 3
        || context.shape().len() != 3
        || query.shape()[0] != context.shape()[0]
    {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidCrossAttentionInput,
        });
    }
    let [batch, query_length, width] = query.shape() else {
        unreachable!("rank checked above")
    };
    if head_dim <= 0 || *width <= 0 || width % head_dim != 0 {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidAttentionHeadDimension,
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

fn feed_forward(cx: Cx, input: &Tensor) -> Result<Tensor> {
    let width = *input.shape().last().ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::MissingFeedForwardDimension,
    })?;
    let hidden = width.checked_mul(4).ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::GegluHiddenWidthOverflow,
    })?;
    let projected = hidden.checked_mul(2).ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::GegluProjectedWidthOverflow,
    })?;
    let projected = cx
        .scope("net")?
        .scope("0")?
        .scope("proj")?
        .linear(projected)
        .apply(input)?;
    let parts = projected.split(projected.shape().len() - 1, &[hidden, hidden])?;
    let gated = parts[0].mul(&parts[1].gelu()?)?;
    Ok(gated.apply(&cx.scope("net")?.layer("2", Linear::new(width))?)?)
}

fn transformer_block(cx: Cx, input: &Tensor, context: &Tensor, head_dim: i64) -> Result<Tensor> {
    let normalized = input.apply(&cx.layer("norm1", LayerNorm::new(1))?)?;
    let attention = {
        let scope = cx.scope("attn1")?;
        cross_attention(scope, &normalized, &normalized, head_dim)?
    };
    let hidden = input.add(&attention)?;
    let normalized = hidden.apply(&cx.layer("norm2", LayerNorm::new(1))?)?;
    let attention = {
        let scope = cx.scope("attn2")?;
        cross_attention(scope, &normalized, context, head_dim)?
    };
    let hidden = hidden.add(&attention)?;
    let normalized = hidden.apply(&cx.layer("norm3", LayerNorm::new(1))?)?;
    let ff = cx.scope("ff")?;
    Ok(hidden.add(&feed_forward(ff, &normalized)?)?)
}

/// Diffusers `Transformer2DModel` for NHWC activations and `[B,T,C]` context.
pub fn spatial_transformer(
    cx: Cx,
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
            kind: ModelDefinitionError::InvalidSpatialTransformerInput,
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
        let scope = cx.scope_path(["transformer_blocks".to_owned(), layer.to_string()])?;
        hidden = transformer_block(scope, &hidden, context, options.head_dim)?;
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

    fn model(cx: Cx) -> Result<Tensor> {
        let image = cx.input(&[2, 16, 12, 32])?;
        let timestep = cx.input(&[2, 128])?;
        let scope = cx.scope("block")?;
        resnet2d(scope, &image, &timestep, Resnet2dOptions::new(64))
    }

    #[test]
    fn timestep_mlp_infers_input_width() {
        let model = Model::new(|cx: Cx| {
            let input = cx.input(&[2, 32])?;
            let scope = cx.scope("time_embedding")?;
            timestep_embedding(scope, &input, 128)
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
        let model = Model::new(|cx: Cx| {
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
        let model = |cx: Cx| {
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
