//! Parameter-effect neural-network building blocks.

use super::*;
use rxla_core::Conv2dOptions;
use snafu::{OptionExt, ensure};

/// A named embedding lookup with an inferred output shape.
#[must_use = "layer builders do nothing until apply is called"]
pub struct Embedding<'a> {
    scope: Scope<'a>,
    vocabulary: i64,
    width: i64,
}

impl Embedding<'_> {
    pub fn apply(mut self, indices: &Tensor) -> Result<Tensor> {
        ensure!(
            indices.dtype() == DType::I32 && self.vocabulary > 0 && self.width > 0,
            InvalidLayerInputSnafu {
                layer: "Embedding",
                requirement: "I32 indices and positive vocabulary/width",
            }
        );
        Ok(self
            .scope
            .param_initialized(
                "weight",
                &[self.vocabulary, self.width],
                Initializer::normal(0.0, 1.0),
            )?
            .take(indices, 0)?)
    }
}

/// A named affine projection whose input width is inferred by [`Linear::apply`].
#[must_use = "layer builders do nothing until apply is called"]
pub struct Linear<'a> {
    scope: Scope<'a>,
    out_features: i64,
    bias: bool,
}

/// Weight-only affine projection with symmetric per-group U8 storage.
///
/// Stored values encode signed INT8 as `value + 128`; `scale` is F32 with one
/// value per output row and input group. Dequantization remains visible in IR
/// so XLA may fuse it into the consuming contraction.
#[must_use = "layer builders do nothing until apply is called"]
pub struct QuantizedLinear<'a> {
    scope: Scope<'a>,
    out_features: i64,
    group_size: i64,
}

impl QuantizedLinear<'_> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        ensure!(
            input.dtype() == DType::F32 && !input.shape().is_empty(),
            InvalidLayerInputSnafu {
                layer: "QuantizedLinear",
                requirement: "an F32 input with rank at least one",
            }
        );
        let in_features = *input.shape().last().expect("rank checked above");
        ensure!(
            self.out_features > 0
                && self.group_size > 0
                && in_features > 0
                && in_features % self.group_size == 0,
            InvalidLayerInputSnafu {
                layer: "QuantizedLinear",
                requirement: "positive widths and a group size dividing the input width",
            }
        );
        let groups = in_features / self.group_size;
        let weight = self
            .scope
            .param_dtype("weight", &[self.out_features, in_features], DType::U8)?
            .cast(DType::F32)?
            .add_scalar(-128.0)?
            .reshape(&[self.out_features, groups, self.group_size])?;
        let scale = self
            .scope
            .param("scale", &[self.out_features, groups])?
            .reshape(&[self.out_features, groups, 1])?
            .broadcast_to(&[self.out_features, groups, self.group_size])?;
        Ok(input.linear(
            &weight
                .mul(&scale)?
                .reshape(&[self.out_features, in_features])?,
            None,
        )?)
    }
}

impl Linear<'_> {
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_linear(&mut self.scope, input, self.out_features, self.bias)
    }
}

/// A named NHWC convolution whose input channels are inferred at its use site.
#[must_use = "layer builders do nothing until apply is called"]
pub struct Conv2d<'a> {
    scope: Scope<'a>,
    out_channels: i64,
    kernel: [i64; 2],
    options: Conv2dOptions,
    bias: bool,
}

impl Conv2d<'_> {
    pub fn options(mut self, options: Conv2dOptions) -> Self {
        self.options = options;
        self
    }

    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_conv2d(
            &mut self.scope,
            input,
            self.out_channels,
            self.kernel,
            self.options,
            self.bias,
        )
    }
}

/// A named GroupNorm operation with optional learned affine parameters.
#[must_use = "layer builders do nothing until apply is called"]
pub struct GroupNorm<'a> {
    scope: Scope<'a>,
    groups: i64,
    epsilon: f32,
    affine: bool,
}

/// Stateful NHWC BatchNorm with inferred channel count.
#[must_use = "layer builders do nothing until apply is called"]
pub struct BatchNorm<'a> {
    scope: Scope<'a>,
    epsilon: f32,
    momentum: f32,
    training: bool,
}

impl BatchNorm<'_> {
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Weight assigned to the current batch statistics in the running EMA.
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }

    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_batch_norm_nhwc(
            &mut self.scope,
            input,
            self.epsilon,
            self.momentum,
            self.training,
        )
    }
}

impl GroupNorm<'_> {
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_group_norm_nhwc(
            &mut self.scope,
            input,
            self.groups,
            self.epsilon,
            self.affine,
        )
    }
}

/// A named LayerNorm operation whose normalized shape is inferred on apply.
#[must_use = "layer builders do nothing until apply is called"]
pub struct LayerNorm<'a> {
    scope: Scope<'a>,
    normalized_rank: usize,
    epsilon: f32,
    affine: bool,
}

/// A named RMSNorm operation whose width is inferred on apply.
#[must_use = "layer builders do nothing until apply is called"]
pub struct RmsNorm<'a> {
    scope: Scope<'a>,
    epsilon: f32,
    zero_centered: bool,
}

impl RmsNorm<'_> {
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Interpret the stored scale as an offset from one, as used by Qwen3.5.
    pub fn zero_centered(mut self, zero_centered: bool) -> Self {
        self.zero_centered = zero_centered;
        self
    }

    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        ensure!(
            input.dtype() == DType::F32 && !input.shape().is_empty(),
            InvalidLayerInputSnafu {
                layer: "RmsNorm",
                requirement: "an F32 input with rank at least one",
            }
        );
        let width = *input.shape().last().expect("rank checked above");
        let weight = self.scope.param_initialized(
            "weight",
            &[width],
            if self.zero_centered {
                Initializer::zeros()
            } else {
                Initializer::ones()
            },
        )?;
        let weight = if self.zero_centered {
            weight.add_scalar(1.0)?
        } else {
            weight
        };
        Ok(input.rms_norm(&weight, self.epsilon)?)
    }
}

impl LayerNorm<'_> {
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_layer_norm(
            &mut self.scope,
            input,
            self.normalized_rank,
            self.epsilon,
            self.affine,
        )
    }
}

/// Builder entry point for one named layer with non-default options.
pub struct Layer<'a> {
    scope: Scope<'a>,
}

impl<'a> Layer<'a> {
    pub fn embedding(self, vocabulary: i64, width: i64) -> Embedding<'a> {
        Embedding {
            scope: self.scope,
            vocabulary,
            width,
        }
    }

    pub fn linear(self, out_features: i64) -> Linear<'a> {
        Linear {
            scope: self.scope,
            out_features,
            bias: true,
        }
    }

    pub fn quantized_linear(self, out_features: i64, group_size: i64) -> QuantizedLinear<'a> {
        QuantizedLinear {
            scope: self.scope,
            out_features,
            group_size,
        }
    }

    pub fn conv2d(self, out_channels: i64, kernel: [i64; 2]) -> Conv2d<'a> {
        Conv2d {
            scope: self.scope,
            out_channels,
            kernel,
            options: Conv2dOptions::default(),
            bias: true,
        }
    }

    pub fn group_norm(self, groups: i64) -> GroupNorm<'a> {
        GroupNorm {
            scope: self.scope,
            groups,
            epsilon: 1e-5,
            affine: true,
        }
    }

    pub fn batch_norm(self) -> BatchNorm<'a> {
        BatchNorm {
            scope: self.scope,
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }

    pub fn layer_norm(self, normalized_rank: usize) -> LayerNorm<'a> {
        LayerNorm {
            scope: self.scope,
            normalized_rank,
            epsilon: 1e-5,
            affine: true,
        }
    }

    pub fn rms_norm(self) -> RmsNorm<'a> {
        RmsNorm {
            scope: self.scope,
            epsilon: 1e-5,
            zero_centered: false,
        }
    }
}

impl Cx {
    /// Enter one named layer builder without changing the effect API surface.
    pub fn layer(&mut self, name: &str) -> Result<Layer<'_>> {
        Ok(Layer {
            scope: self.scope(name)?,
        })
    }
}

/// Apply an F32 affine projection using parameters declared at the current
/// lexical scope. The input feature dimension is inferred from `input`.
fn apply_linear(cx: &mut Cx, input: &Tensor, out_features: i64, bias: bool) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32,
        InvalidLayerInputSnafu {
            layer: "Linear",
            requirement: "an F32 computation tensor",
        }
    );
    let in_features = *input.shape().last().context(InvalidLayerInputSnafu {
        layer: "Linear",
        requirement: "an input with at least one dimension",
    })?;
    ensure!(
        in_features > 0 && out_features > 0,
        InvalidLayerInputSnafu {
            layer: "Linear",
            requirement: "positive input and output feature dimensions",
        }
    );
    let weight = cx.param_initialized(
        "weight",
        &[out_features, in_features],
        Initializer::kaiming_uniform(),
    )?;
    let bias_bound = (1.0 / in_features as f32).sqrt();
    let bias = bias
        .then(|| {
            cx.param_initialized(
                "bias",
                &[out_features],
                Initializer::uniform(-bias_bound, bias_bound),
            )
        })
        .transpose()?;
    Ok(input.linear(&weight, bias.as_ref())?)
}

/// Apply an NHWC convolution with checkpoint-native OIHW parameters at the
/// current lexical scope. Input channels are inferred from `input`.
fn apply_conv2d(
    cx: &mut Cx,
    input: &Tensor,
    out_channels: i64,
    kernel: [i64; 2],
    options: Conv2dOptions,
    bias: bool,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32 && input.shape().len() == 4,
        InvalidLayerInputSnafu {
            layer: "Conv2d",
            requirement: "a rank-four F32 NHWC tensor",
        }
    );
    let in_channels = input.shape()[3];
    ensure!(
        in_channels > 0
            && out_channels > 0
            && kernel.iter().all(|&dim| dim > 0)
            && options.groups > 0
            && in_channels % options.groups == 0,
        InvalidLayerInputSnafu {
            layer: "Conv2d",
            requirement: "positive channels/kernel and compatible groups",
        }
    );
    let weight = cx.param_initialized(
        "weight",
        &[
            out_channels,
            in_channels / options.groups,
            kernel[0],
            kernel[1],
        ],
        Initializer::kaiming_uniform(),
    )?;
    let output = input.conv2d_oihw(&weight, options)?;
    if !bias {
        return Ok(output);
    }
    let fan_in = (in_channels / options.groups) * kernel[0] * kernel[1];
    let bound = (1.0 / fan_in as f32).sqrt();
    let bias =
        cx.param_initialized("bias", &[out_channels], Initializer::uniform(-bound, bound))?;
    Ok(output.add(&bias.broadcast_to(output.shape())?)?)
}

/// Apply GroupNorm to an NHWC activation using `[C]` affine parameters at
/// the current lexical scope. Channels are inferred from the input tensor.
fn apply_group_norm_nhwc(
    cx: &mut Cx,
    input: &Tensor,
    groups: i64,
    epsilon: f32,
    affine: bool,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32 && input.shape().len() == 4,
        InvalidLayerInputSnafu {
            layer: "GroupNorm",
            requirement: "a rank-four F32 NHWC tensor",
        }
    );
    let channels = input.shape()[3];
    ensure!(
        channels > 0 && groups > 0 && channels % groups == 0,
        InvalidLayerInputSnafu {
            layer: "GroupNorm",
            requirement: "positive groups dividing channels",
        }
    );
    let (weight, bias) = if affine {
        (
            Some(cx.param_initialized("weight", &[channels], Initializer::ones())?),
            Some(cx.param_initialized("bias", &[channels], Initializer::zeros())?),
        )
    } else {
        (None, None)
    };
    Ok(input
        .transpose(&[0, 3, 1, 2])?
        .group_norm(groups, weight.as_ref(), bias.as_ref(), epsilon)?
        .transpose(&[0, 2, 3, 1])?)
}

fn apply_batch_norm_nhwc(
    cx: &mut Cx,
    input: &Tensor,
    epsilon: f32,
    momentum: f32,
    training: bool,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32
            && input.shape().len() == 4
            && epsilon.is_finite()
            && epsilon > 0.0
            && momentum.is_finite()
            && (0.0..=1.0).contains(&momentum),
        InvalidLayerInputSnafu {
            layer: "BatchNorm",
            requirement: "rank-four F32 NHWC input, positive epsilon, and momentum in [0, 1]",
        }
    );
    let channels = input.shape()[3];
    let weight = cx.param_initialized("weight", &[channels], Initializer::ones())?;
    let bias = cx.param_initialized("bias", &[channels], Initializer::zeros())?;
    let running_mean = cx.state("running_mean", &[channels], DType::F32)?;
    let running_variance = cx.state_initialized(
        "running_variance",
        &[channels],
        DType::F32,
        Initializer::ones(),
    )?;
    if !training {
        return Ok(input.batch_norm_inference(
            3,
            &running_mean.read(cx)?,
            &running_variance.read(cx)?,
            &weight,
            &bias,
            epsilon,
        )?);
    }

    let batch = input.batch_norm_training(3, &weight, &bias, epsilon)?;
    let retain = 1.0 - momentum;
    let next_mean = running_mean
        .read(cx)?
        .mul_scalar(retain)?
        .add(&batch.mean.mul_scalar(momentum)?)?;
    let next_variance = running_variance
        .read(cx)?
        .mul_scalar(retain)?
        .add(&batch.variance.mul_scalar(momentum)?)?;
    running_mean.write(cx, &next_mean)?;
    running_variance.write(cx, &next_variance)?;
    Ok(batch.output)
}

/// Apply LayerNorm over the final `normalized_rank` dimensions, inferring
/// affine parameter shapes directly from the input tensor.
fn apply_layer_norm(
    cx: &mut Cx,
    input: &Tensor,
    normalized_rank: usize,
    epsilon: f32,
    affine: bool,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32
            && normalized_rank > 0
            && normalized_rank <= input.shape().len(),
        InvalidLayerInputSnafu {
            layer: "LayerNorm",
            requirement: "F32 input and a valid positive normalized rank",
        }
    );
    let normalized_shape = &input.shape()[input.shape().len() - normalized_rank..];
    let (weight, bias) = if affine {
        (
            Some(cx.param_initialized("weight", normalized_shape, Initializer::ones())?),
            Some(cx.param_initialized("bias", normalized_shape, Initializer::zeros())?),
        )
    } else {
        (None, None)
    };
    Ok(input.layer_norm(normalized_shape, weight.as_ref(), bias.as_ref(), epsilon)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{apply, init};

    fn quantized_projection(cx: &mut Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 64])?;
        cx.layer("projection")?
            .quantized_linear(32, 32)
            .apply(&input)
    }

    #[test]
    fn quantized_linear_keeps_integer_storage_in_the_model_abi() {
        let (schema, output) = init(quantized_projection).unwrap();
        assert_eq!(output.shape(), [2, 32]);
        assert_eq!(schema.get("projection.weight").unwrap().dtype(), DType::U8);
        assert_eq!(schema.get("projection.scale").unwrap().shape(), [32, 2]);
        apply(&schema, quantized_projection)
            .unwrap()
            .prepare()
            .unwrap();
    }

    #[test]
    fn direct_named_ops_preserve_scoped_parameter_identity() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 8])?;
            let encoder = cx.layer("encoder")?.linear(4);
            let hidden = encoder.apply(&input)?;
            cx.layer("head")?.linear(3).apply(&hidden)
        })
        .unwrap();

        assert_eq!(output.shape(), [2, 3]);
        assert_eq!(schema.get("encoder.weight").unwrap().shape(), [4, 8]);
        assert_eq!(schema.get("encoder.bias").unwrap().shape(), [4]);
        assert_eq!(schema.get("head.weight").unwrap().shape(), [3, 4]);
        assert_eq!(schema.get("head.bias").unwrap().shape(), [3]);
    }
}
