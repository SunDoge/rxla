//! Parameter-effect neural-network building blocks.

use super::*;
use derive_setters::Setters;
use rxla_core::Conv2dOptions;
use snafu::{OptionExt, ensure};

/// Logical image activation layout accepted by spatial neural-network layers.
/// Checkpoint convolution kernels remain OIHW in either layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageLayout {
    Nchw,
    #[default]
    Nhwc,
}

/// A stateless layer configuration interpreted inside a named effect scope.
pub trait Layer: Sized {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor>;

    fn named(self, name: impl Into<String>) -> NamedLayer<Self> {
        NamedLayer {
            name: name.into(),
            layer: self,
        }
    }
}

/// A reusable layer configuration with a stable name relative to an effect
/// scope. It holds no parameters and does not borrow the context.
pub struct NamedLayer<L> {
    name: String,
    layer: L,
}

impl<L: Layer> NamedLayer<L> {
    pub fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        let mut scope = cx.scope(&self.name)?;
        self.layer.apply(&mut scope, input)
    }
}

/// A named embedding lookup with an inferred output shape.
#[must_use = "layer builders do nothing until apply is called"]
pub struct Embedding {
    vocabulary: i64,
    width: i64,
}

impl Embedding {
    pub fn new(vocabulary: i64, width: i64) -> Self {
        Self { vocabulary, width }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, indices: &Tensor) -> Result<Tensor> {
        ensure!(
            indices.dtype() == DType::I32 && self.vocabulary > 0 && self.width > 0,
            InvalidLayerInputSnafu {
                layer: "Embedding",
                requirement: "I32 indices and positive vocabulary/width",
            }
        );
        Ok(cx
            .scope(name)?
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
#[derive(Setters)]
#[setters(generate = false)]
pub struct Linear {
    out_features: i64,
    #[setters(generate)]
    bias: bool,
}

/// Weight-only affine projection with symmetric per-group U8 storage.
///
/// Stored values encode signed INT8 as `value + 128`; `scale` is F32 with one
/// value per output row and input group. Dequantization remains visible in IR
/// so XLA may fuse it into the consuming contraction.
#[must_use = "layer builders do nothing until apply is called"]
pub struct QuantizedLinear {
    out_features: i64,
    group_size: i64,
}

impl QuantizedLinear {
    pub fn new(out_features: i64, group_size: i64) -> Self {
        Self {
            out_features,
            group_size,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        let mut scope = cx.scope(name)?;
        apply_quantized_linear(&mut scope, input, &self)
    }
}

fn apply_quantized_linear(
    scope: &mut Cx,
    input: &Tensor,
    layer: &QuantizedLinear,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32 && !input.shape().is_empty(),
        InvalidLayerInputSnafu {
            layer: "QuantizedLinear",
            requirement: "an F32 input with rank at least one",
        }
    );
    let in_features = *input.shape().last().expect("rank checked above");
    ensure!(
        layer.out_features > 0
            && layer.group_size > 0
            && in_features > 0
            && in_features % layer.group_size == 0,
        InvalidLayerInputSnafu {
            layer: "QuantizedLinear",
            requirement: "positive widths and a group size dividing the input width",
        }
    );
    let groups = in_features / layer.group_size;
    let weight = scope
        .param_dtype("weight", &[layer.out_features, in_features], DType::U8)?
        .cast(DType::F32)?
        .add_scalar(-128.0)?
        .reshape(&[layer.out_features, groups, layer.group_size])?;
    let scale = scope
        .param("scale", &[layer.out_features, groups])?
        .reshape(&[layer.out_features, groups, 1])?
        .broadcast_to(&[layer.out_features, groups, layer.group_size])?;
    Ok(input.linear(
        &weight
            .mul(&scale)?
            .reshape(&[layer.out_features, in_features])?,
        None,
    )?)
}

impl Linear {
    pub fn new(out_features: i64) -> Self {
        Self {
            out_features,
            bias: true,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        let mut scope = cx.scope(name)?;
        apply_linear(&mut scope, input, self.out_features, self.bias)
    }
}

/// A named NHWC convolution whose input channels are inferred at its use site.
#[must_use = "layer builders do nothing until apply is called"]
#[derive(Setters)]
#[setters(generate = false)]
pub struct Conv2d {
    out_channels: i64,
    kernel: [i64; 2],
    #[setters(generate)]
    options: Conv2dOptions,
    #[setters(generate)]
    bias: bool,
    #[setters(generate)]
    layout: ImageLayout,
}

impl Conv2d {
    pub fn new(out_channels: i64, kernel: [i64; 2]) -> Self {
        Self {
            out_channels,
            kernel,
            options: Conv2dOptions::default(),
            bias: true,
            layout: ImageLayout::Nhwc,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        let mut scope = cx.scope(name)?;
        apply_conv2d(
            &mut scope,
            input,
            self.out_channels,
            self.kernel,
            self.options,
            self.bias,
            self.layout,
        )
    }
}

/// A named GroupNorm operation with optional learned affine parameters.
#[must_use = "layer builders do nothing until apply is called"]
#[derive(Setters)]
#[setters(generate = false)]
pub struct GroupNorm {
    groups: i64,
    #[setters(generate)]
    epsilon: f32,
    #[setters(generate)]
    affine: bool,
    #[setters(generate)]
    layout: ImageLayout,
}

/// Stateful NHWC BatchNorm with inferred channel count.
#[must_use = "layer builders do nothing until apply is called"]
#[derive(Setters)]
#[setters(generate = false)]
pub struct BatchNorm {
    #[setters(generate)]
    epsilon: f32,
    #[setters(generate)]
    momentum: f32,
    #[setters(generate)]
    layout: ImageLayout,
}

impl BatchNorm {
    pub fn new() -> Self {
        Self {
            epsilon: 1e-5,
            momentum: 0.1,
            layout: ImageLayout::Nhwc,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        let training = cx.mode() == ExecutionMode::Training;
        let mut scope = cx.scope(name)?;
        apply_batch_norm(
            &mut scope,
            input,
            self.epsilon,
            self.momentum,
            training,
            self.layout,
        )
    }
}

impl Default for BatchNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupNorm {
    pub fn new(groups: i64) -> Self {
        Self {
            groups,
            epsilon: 1e-5,
            affine: true,
            layout: ImageLayout::Nhwc,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        let mut scope = cx.scope(name)?;
        apply_group_norm(
            &mut scope,
            input,
            self.groups,
            self.epsilon,
            self.affine,
            self.layout,
        )
    }
}

/// A named LayerNorm operation whose normalized shape is inferred on apply.
#[must_use = "layer builders do nothing until apply is called"]
#[derive(Setters)]
#[setters(generate = false)]
pub struct LayerNorm {
    normalized_rank: usize,
    #[setters(generate)]
    epsilon: f32,
    #[setters(generate)]
    affine: bool,
}

/// A named RMSNorm operation whose width is inferred on apply.
#[must_use = "layer builders do nothing until apply is called"]
#[derive(Setters)]
#[setters(generate = false)]
pub struct RmsNorm {
    #[setters(generate)]
    epsilon: f32,
    #[setters(generate)]
    zero_centered: bool,
}

impl RmsNorm {
    pub fn new() -> Self {
        Self {
            epsilon: 1e-5,
            zero_centered: false,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        ensure!(
            input.dtype() == DType::F32 && !input.shape().is_empty(),
            InvalidLayerInputSnafu {
                layer: "RmsNorm",
                requirement: "an F32 input with rank at least one",
            }
        );
        let width = *input.shape().last().expect("rank checked above");
        let weight = cx.scope(name)?.param_initialized(
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

impl Default for RmsNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerNorm {
    pub fn new(normalized_rank: usize) -> Self {
        Self {
            normalized_rank,
            epsilon: 1e-5,
            affine: true,
        }
    }

    pub fn apply(self, cx: &mut Cx, name: &str, input: &Tensor) -> Result<Tensor> {
        let mut scope = cx.scope(name)?;
        apply_layer_norm(
            &mut scope,
            input,
            self.normalized_rank,
            self.epsilon,
            self.affine,
        )
    }
}

impl Layer for Linear {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        apply_linear(cx, input, self.out_features, self.bias)
    }
}

impl Layer for Conv2d {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        apply_conv2d(
            cx,
            input,
            self.out_channels,
            self.kernel,
            self.options,
            self.bias,
            self.layout,
        )
    }
}

impl Layer for GroupNorm {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        apply_group_norm(
            cx,
            input,
            self.groups,
            self.epsilon,
            self.affine,
            self.layout,
        )
    }
}

impl Layer for BatchNorm {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        let training = cx.mode() == ExecutionMode::Training;
        apply_batch_norm(
            cx,
            input,
            self.epsilon,
            self.momentum,
            training,
            self.layout,
        )
    }
}

impl Layer for LayerNorm {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        apply_layer_norm(cx, input, self.normalized_rank, self.epsilon, self.affine)
    }
}

impl Layer for QuantizedLinear {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        apply_quantized_linear(cx, input, self)
    }
}

impl Layer for Embedding {
    fn apply(&self, cx: &mut Cx, indices: &Tensor) -> Result<Tensor> {
        ensure!(
            indices.dtype() == DType::I32 && self.vocabulary > 0 && self.width > 0,
            InvalidLayerInputSnafu {
                layer: "Embedding",
                requirement: "I32 indices and positive vocabulary/width",
            }
        );
        Ok(cx
            .param_initialized(
                "weight",
                &[self.vocabulary, self.width],
                Initializer::normal(0.0, 1.0),
            )?
            .take(indices, 0)?)
    }
}

impl Layer for RmsNorm {
    fn apply(&self, cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
        ensure!(
            input.dtype() == DType::F32 && !input.shape().is_empty(),
            InvalidLayerInputSnafu {
                layer: "RmsNorm",
                requirement: "an F32 input with rank at least one",
            }
        );
        let width = *input.shape().last().expect("rank checked above");
        let weight = cx.param_initialized(
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

impl Cx {
    pub fn apply<L: Layer>(&mut self, name: &str, layer: L, input: &Tensor) -> Result<Tensor> {
        let mut scope = self.scope(name)?;
        Layer::apply(&layer, &mut scope, input)
    }
}

/// Compatibility adapter for the former scope-first spelling.
/// New model code should construct a layer configuration and call
/// `layer.apply(cx, name, input)` directly.
#[doc(hidden)]
pub struct ScopedLayer<'a, L> {
    scope: Scope<'a>,
    layer: L,
}

macro_rules! scoped_setters {
    ($layer:ty, $($name:ident : $ty:ty),* $(,)?) => {
        impl<'a> ScopedLayer<'a, $layer> {
            $(
                pub fn $name(mut self, value: $ty) -> Self {
                    self.layer = self.layer.$name(value);
                    self
                }
            )*
        }
    };
}

scoped_setters!(Linear, bias: bool);
scoped_setters!(Conv2d, options: Conv2dOptions, bias: bool, layout: ImageLayout);
scoped_setters!(GroupNorm, epsilon: f32, affine: bool, layout: ImageLayout);
scoped_setters!(BatchNorm, epsilon: f32, momentum: f32, layout: ImageLayout);
scoped_setters!(LayerNorm, epsilon: f32, affine: bool);
scoped_setters!(RmsNorm, epsilon: f32, zero_centered: bool);

// The adapter cannot call the public named API because that would introduce an
// extra path component. These implementations intentionally execute the same
// layer helpers in the scope already acquired by the old spelling.
impl ScopedLayer<'_, Linear> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_linear(
            &mut self.scope,
            input,
            self.layer.out_features,
            self.layer.bias,
        )
    }
}

impl ScopedLayer<'_, Conv2d> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_conv2d(
            &mut self.scope,
            input,
            self.layer.out_channels,
            self.layer.kernel,
            self.layer.options,
            self.layer.bias,
            self.layer.layout,
        )
    }
}

impl ScopedLayer<'_, GroupNorm> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_group_norm(
            &mut self.scope,
            input,
            self.layer.groups,
            self.layer.epsilon,
            self.layer.affine,
            self.layer.layout,
        )
    }
}

impl ScopedLayer<'_, BatchNorm> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        let training = self.scope.mode() == ExecutionMode::Training;
        apply_batch_norm(
            &mut self.scope,
            input,
            self.layer.epsilon,
            self.layer.momentum,
            training,
            self.layer.layout,
        )
    }
}

impl ScopedLayer<'_, LayerNorm> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_layer_norm(
            &mut self.scope,
            input,
            self.layer.normalized_rank,
            self.layer.epsilon,
            self.layer.affine,
        )
    }
}

impl ScopedLayer<'_, RmsNorm> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        let width = *input.shape().last().context(InvalidLayerInputSnafu {
            layer: "RmsNorm",
            requirement: "an F32 input with rank at least one",
        })?;
        let weight = self.scope.param_initialized(
            "weight",
            &[width],
            if self.layer.zero_centered {
                Initializer::zeros()
            } else {
                Initializer::ones()
            },
        )?;
        let weight = if self.layer.zero_centered {
            weight.add_scalar(1.0)?
        } else {
            weight
        };
        Ok(input.rms_norm(&weight, self.layer.epsilon)?)
    }
}

impl ScopedLayer<'_, Embedding> {
    pub fn apply(mut self, indices: &Tensor) -> Result<Tensor> {
        ensure!(
            indices.dtype() == DType::I32 && self.layer.vocabulary > 0 && self.layer.width > 0,
            InvalidLayerInputSnafu {
                layer: "Embedding",
                requirement: "I32 indices and positive vocabulary/width",
            }
        );
        Ok(self
            .scope
            .param_initialized(
                "weight",
                &[self.layer.vocabulary, self.layer.width],
                Initializer::normal(0.0, 1.0),
            )?
            .take(indices, 0)?)
    }
}

impl ScopedLayer<'_, QuantizedLinear> {
    pub fn apply(mut self, input: &Tensor) -> Result<Tensor> {
        apply_quantized_linear(&mut self.scope, input, &self.layer)
    }
}

impl<'a> Scope<'a> {
    pub fn embedding(self, vocabulary: i64, width: i64) -> ScopedLayer<'a, Embedding> {
        ScopedLayer {
            scope: self,
            layer: Embedding::new(vocabulary, width),
        }
    }

    pub fn linear(self, out_features: i64) -> ScopedLayer<'a, Linear> {
        ScopedLayer {
            scope: self,
            layer: Linear::new(out_features),
        }
    }

    pub fn quantized_linear(
        self,
        out_features: i64,
        group_size: i64,
    ) -> ScopedLayer<'a, QuantizedLinear> {
        ScopedLayer {
            scope: self,
            layer: QuantizedLinear::new(out_features, group_size),
        }
    }

    pub fn conv2d(self, out_channels: i64, kernel: [i64; 2]) -> ScopedLayer<'a, Conv2d> {
        ScopedLayer {
            scope: self,
            layer: Conv2d::new(out_channels, kernel),
        }
    }

    pub fn group_norm(self, groups: i64) -> ScopedLayer<'a, GroupNorm> {
        ScopedLayer {
            scope: self,
            layer: GroupNorm::new(groups),
        }
    }

    pub fn batch_norm(self) -> ScopedLayer<'a, BatchNorm> {
        ScopedLayer {
            scope: self,
            layer: BatchNorm::new(),
        }
    }

    pub fn layer_norm(self, normalized_rank: usize) -> ScopedLayer<'a, LayerNorm> {
        ScopedLayer {
            scope: self,
            layer: LayerNorm::new(normalized_rank),
        }
    }

    pub fn rms_norm(self) -> ScopedLayer<'a, RmsNorm> {
        ScopedLayer {
            scope: self,
            layer: RmsNorm::new(),
        }
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
    layout: ImageLayout,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32 && input.shape().len() == 4,
        InvalidLayerInputSnafu {
            layer: "Conv2d",
            requirement: "a rank-four F32 image tensor",
        }
    );
    let in_channels = input.shape()[match layout {
        ImageLayout::Nchw => 1,
        ImageLayout::Nhwc => 3,
    }];
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
    let input = match layout {
        ImageLayout::Nchw => input.transpose(&[0, 2, 3, 1])?,
        ImageLayout::Nhwc => input.clone(),
    };
    let output = input.conv2d_oihw(&weight, options)?;
    if !bias {
        return match layout {
            ImageLayout::Nchw => Ok(output.transpose(&[0, 3, 1, 2])?),
            ImageLayout::Nhwc => Ok(output),
        };
    }
    let fan_in = (in_channels / options.groups) * kernel[0] * kernel[1];
    let bound = (1.0 / fan_in as f32).sqrt();
    let bias =
        cx.param_initialized("bias", &[out_channels], Initializer::uniform(-bound, bound))?;
    let output = output.add(&bias.broadcast_to(output.shape())?)?;
    match layout {
        ImageLayout::Nchw => Ok(output.transpose(&[0, 3, 1, 2])?),
        ImageLayout::Nhwc => Ok(output),
    }
}

/// Apply GroupNorm using `[C]` affine parameters at the current lexical scope.
/// Channels are inferred according to the declared logical image layout.
fn apply_group_norm(
    cx: &mut Cx,
    input: &Tensor,
    groups: i64,
    epsilon: f32,
    affine: bool,
    layout: ImageLayout,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32 && input.shape().len() == 4,
        InvalidLayerInputSnafu {
            layer: "GroupNorm",
            requirement: "a rank-four F32 image tensor",
        }
    );
    let channels = input.shape()[match layout {
        ImageLayout::Nchw => 1,
        ImageLayout::Nhwc => 3,
    }];
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
    let input = match layout {
        ImageLayout::Nchw => input.clone(),
        ImageLayout::Nhwc => input.transpose(&[0, 3, 1, 2])?,
    };
    let output = input.group_norm(groups, weight.as_ref(), bias.as_ref(), epsilon)?;
    match layout {
        ImageLayout::Nchw => Ok(output),
        ImageLayout::Nhwc => Ok(output.transpose(&[0, 2, 3, 1])?),
    }
}

fn apply_batch_norm(
    cx: &mut Cx,
    input: &Tensor,
    epsilon: f32,
    momentum: f32,
    training: bool,
    layout: ImageLayout,
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
            requirement: "rank-four F32 image input, positive epsilon, and momentum in [0, 1]",
        }
    );
    let channels = input.shape()[match layout {
        ImageLayout::Nchw => 1,
        ImageLayout::Nhwc => 3,
    }];
    let input = match layout {
        ImageLayout::Nchw => input.transpose(&[0, 2, 3, 1])?,
        ImageLayout::Nhwc => input.clone(),
    };
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
        let output = input.batch_norm_inference(
            3,
            &running_mean.read(cx)?,
            &running_variance.read(cx)?,
            &weight,
            &bias,
            epsilon,
        )?;
        return match layout {
            ImageLayout::Nchw => Ok(output.transpose(&[0, 3, 1, 2])?),
            ImageLayout::Nhwc => Ok(output),
        };
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
    match layout {
        ImageLayout::Nchw => Ok(batch.output.transpose(&[0, 3, 1, 2])?),
        ImageLayout::Nhwc => Ok(batch.output),
    }
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
        cx.scope("projection")?
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
    fn scoped_builders_preserve_parameter_identity() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 8])?;
            let encoder = cx.scope("encoder")?.linear(4);
            let hidden = encoder.apply(&input)?;
            cx.scope("head")?.linear(3).apply(&hidden)
        })
        .unwrap();

        assert_eq!(output.shape(), [2, 3]);
        assert_eq!(schema.get("encoder.weight").unwrap().shape(), [4, 8]);
        assert_eq!(schema.get("encoder.bias").unwrap().shape(), [4]);
        assert_eq!(schema.get("head.weight").unwrap().shape(), [3, 4]);
        assert_eq!(schema.get("head.bias").unwrap().shape(), [3]);
    }

    #[test]
    fn layer_configs_apply_named_effects_without_a_builder_type() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 8])?;
            let encoder = Linear::new(4).bias(false).named("encoder");
            let head = Linear::new(3).named("head");
            let hidden = encoder.apply(cx, &input)?;
            head.apply(cx, &hidden)
        })
        .unwrap();

        assert_eq!(output.shape(), [2, 3]);
        assert_eq!(schema.get("encoder.weight").unwrap().shape(), [4, 8]);
        assert!(schema.get("encoder.bias").is_none());
        assert_eq!(schema.get("head.weight").unwrap().shape(), [3, 4]);
        assert_eq!(schema.get("head.bias").unwrap().shape(), [3]);
    }

    #[test]
    fn named_layers_can_be_reused_without_owning_parameters() {
        let (schema, outputs) = init(|cx| {
            let first = cx.input(&[2, 8])?;
            let second = cx.input(&[2, 8])?;
            let projection = Linear::new(4).named("projection");
            Ok::<_, Error>([
                projection.apply(cx, &first)?,
                projection.apply(cx, &second)?,
            ])
        })
        .unwrap();

        assert_eq!(outputs[0].shape(), [2, 4]);
        assert_eq!(outputs[1].shape(), [2, 4]);
        assert_eq!(schema.parameters().len(), 2);
        assert_eq!(schema.get("projection.weight").unwrap().shape(), [4, 8]);
    }

    #[test]
    fn spatial_builders_accept_checkpoint_native_nchw_activations() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 4, 16, 12])?;
            let convolved = cx
                .scope("conv")?
                .conv2d(8, [3, 3])
                .layout(ImageLayout::Nchw)
                .apply(&input)?;
            cx.scope("norm")?
                .group_norm(4)
                .layout(ImageLayout::Nchw)
                .apply(&convolved)
        })
        .unwrap();

        assert_eq!(output.shape(), [2, 8, 14, 10]);
        assert_eq!(schema.get("conv.weight").unwrap().shape(), [8, 4, 3, 3]);
        assert_eq!(schema.get("norm.weight").unwrap().shape(), [8]);
    }

    #[test]
    fn downstream_vocabulary_can_be_an_extension_trait_on_scope() {
        trait ProjectionExt<'a> {
            fn projection(self, width: i64) -> ScopedLayer<'a, Linear>;
        }

        impl<'a> ProjectionExt<'a> for Scope<'a> {
            fn projection(self, width: i64) -> ScopedLayer<'a, Linear> {
                self.linear(width).bias(false)
            }
        }

        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 8])?;
            cx.scope("adapter")?.projection(4).apply(&input)
        })
        .unwrap();

        assert_eq!(output.shape(), [2, 4]);
        assert_eq!(schema.get("adapter.weight").unwrap().shape(), [4, 8]);
        assert!(schema.get("adapter.bias").is_none());
    }
}
