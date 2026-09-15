//! Functional Diffusers-compatible AutoencoderKL decoder.

use super::*;

fn convolution(kernel: i64) -> Conv2dOptions {
    Conv2dOptions {
        padding: [[kernel / 2, kernel / 2]; 2],
        ..Default::default()
    }
}

#[derive(Clone, Debug)]
pub struct AutoencoderKlDecoderConfig {
    latent_channels: i64,
    output_channels: i64,
    block_channels: Vec<i64>,
    layers_per_block: usize,
    norm_groups: i64,
    norm_epsilon: f32,
    scaling_factor: f32,
}

impl AutoencoderKlDecoderConfig {
    pub fn tiny() -> Self {
        Self {
            latent_channels: 4,
            output_channels: 3,
            block_channels: vec![32, 64],
            layers_per_block: 1,
            norm_groups: 32,
            norm_epsilon: 1e-6,
            scaling_factor: 0.18215,
        }
    }

    pub fn stable_diffusion() -> Self {
        Self {
            latent_channels: 4,
            output_channels: 3,
            block_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_groups: 32,
            norm_epsilon: 1e-6,
            scaling_factor: 0.18215,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.latent_channels <= 0
            || self.output_channels <= 0
            || self.block_channels.is_empty()
            || self.layers_per_block == 0
            || self.norm_groups <= 0
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || !self.scaling_factor.is_finite()
            || self.scaling_factor <= 0.0
            || self
                .block_channels
                .iter()
                .any(|&channels| channels <= 0 || channels % self.norm_groups != 0)
        {
            return Err(Error::InvalidDefinition {
                message: "invalid AutoencoderKL decoder configuration".into(),
            });
        }
        Ok(())
    }
}

fn indexed_scope<T>(
    cx: &mut Cx,
    collection: &str,
    index: usize,
    build: impl FnOnce(&mut Cx) -> Result<T>,
) -> Result<T> {
    cx.scope(collection, |cx| cx.scope(&index.to_string(), build))
}

fn resnet(
    cx: &mut Cx,
    input: &Tensor,
    output_channels: i64,
    groups: i64,
    epsilon: f32,
) -> Result<Tensor> {
    let normalized = cx
        .named("norm1")?
        .group_norm(groups)
        .epsilon(epsilon)
        .apply(input)?;
    let hidden = cx
        .named("conv1")?
        .conv2d(output_channels, [3, 3])
        .options(convolution(3))
        .apply(&normalized.silu()?)?;
    let normalized = cx
        .named("norm2")?
        .group_norm(groups)
        .epsilon(epsilon)
        .apply(&hidden)?;
    let hidden = cx
        .named("conv2")?
        .conv2d(output_channels, [3, 3])
        .options(convolution(3))
        .apply(&normalized.silu()?)?;
    let residual = if input.shape()[3] == output_channels {
        input.clone()
    } else {
        cx.named("conv_shortcut")?
            .conv2d(output_channels, [1, 1])
            .apply(input)?
    };
    Ok(hidden.add(&residual)?)
}

fn attention(cx: &mut Cx, input: &Tensor, groups: i64, epsilon: f32) -> Result<Tensor> {
    let [batch, height, width, channels] = input.shape() else {
        return Err(Error::InvalidDefinition {
            message: "VAE attention expects NHWC input".into(),
        });
    };
    let hidden = cx
        .named("group_norm")?
        .group_norm(groups)
        .epsilon(epsilon)
        .apply(input)?
        .reshape(&[*batch, height * width, *channels])?;
    let q = cx.named("to_q")?.linear(*channels).apply(&hidden)?;
    let k = cx
        .named("to_k")?
        .linear(*channels)
        .apply(&hidden)?
        .transpose(&[0, 2, 1])?;
    let v = cx.named("to_v")?.linear(*channels).apply(&hidden)?;
    let attended = q
        .matmul(&k)?
        .mul_scalar(1.0 / (*channels as f32).sqrt())?
        .softmax(2)?
        .matmul(&v)?;
    Ok(cx
        .scope("to_out", |cx| {
            cx.named("0")?.linear(*channels).apply(&attended)
        })?
        .reshape(input.shape())?
        .add(input)?)
}

/// Decode an NHWC latent into an NHWC image while interpreting parameter
/// declarations from the surrounding [`Cx`].
pub fn autoencoder_kl_decoder(
    cx: &mut Cx,
    latent: &Tensor,
    config: &AutoencoderKlDecoderConfig,
) -> Result<Tensor> {
    config.validate()?;
    if latent.shape().len() != 4 || latent.shape()[3] != config.latent_channels {
        return Err(Error::InvalidDefinition {
            message: "VAE decoder latent shape does not match its configuration".into(),
        });
    }
    let deepest = *config
        .block_channels
        .last()
        .ok_or_else(|| Error::InvalidDefinition {
            message: "validated VAE block channels are nonempty".into(),
        })?;
    let mut hidden = latent.mul_scalar(1.0 / config.scaling_factor)?;
    hidden = cx
        .named("post_quant_conv")?
        .conv2d(config.latent_channels, [1, 1])
        .apply(&hidden)?;
    hidden = cx.scope("decoder", |cx| {
        cx.named("conv_in")?
            .conv2d(deepest, [3, 3])
            .options(convolution(3))
            .apply(&hidden)
    })?;
    hidden = cx.scope("decoder", |cx| {
        cx.scope("mid_block", |cx| {
            let first = indexed_scope(cx, "resnets", 0, |cx| {
                resnet(
                    cx,
                    &hidden,
                    deepest,
                    config.norm_groups,
                    config.norm_epsilon,
                )
            })?;
            let attended = indexed_scope(cx, "attentions", 0, |cx| {
                attention(cx, &first, config.norm_groups, config.norm_epsilon)
            })?;
            indexed_scope(cx, "resnets", 1, |cx| {
                resnet(
                    cx,
                    &attended,
                    deepest,
                    config.norm_groups,
                    config.norm_epsilon,
                )
            })
        })
    })?;

    let stages = config.block_channels.len();
    for (stage, &output_channels) in config.block_channels.iter().rev().enumerate() {
        hidden = cx.scope("decoder", |cx| {
            indexed_scope(cx, "up_blocks", stage, |cx| {
                let mut value = hidden;
                for layer in 0..=config.layers_per_block {
                    value = indexed_scope(cx, "resnets", layer, |cx| {
                        resnet(
                            cx,
                            &value,
                            output_channels,
                            config.norm_groups,
                            config.norm_epsilon,
                        )
                    })?;
                }
                if stage + 1 < stages {
                    value = cx.scope("upsamplers", |cx| {
                        cx.scope("0", |cx| {
                            cx.named("conv")?
                                .conv2d(output_channels, [3, 3])
                                .options(convolution(3))
                                .apply(&value.upsample_nearest2d([2, 2])?)
                        })
                    })?;
                }
                Ok(value)
            })
        })?;
    }
    let normalized = cx.scope("decoder", |cx| {
        cx.named("conv_norm_out")?
            .group_norm(config.norm_groups)
            .epsilon(config.norm_epsilon)
            .apply(&hidden)
    })?;
    cx.scope("decoder", |cx| {
        cx.named("conv_out")?
            .conv2d(config.output_channels, [3, 3])
            .options(convolution(3))
            .apply(&normalized.silu()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::{apply, init};

    fn tiny(cx: &mut Cx) -> Result<Tensor> {
        let latent = cx.input(&[1, 8, 6, 4])?;
        autoencoder_kl_decoder(cx, &latent, &AutoencoderKlDecoderConfig::tiny())
    }

    #[test]
    fn tiny_decoder_has_stable_schema_and_lowers() {
        let (schema, output) = init(tiny).unwrap();
        assert_eq!(output.shape(), [1, 16, 12, 3]);
        assert_eq!(schema.parameters().len(), 70);
        assert_eq!(
            schema.get("decoder.conv_in.weight").unwrap().shape(),
            [64, 4, 3, 3]
        );
        assert!(schema.get("decoder.conv_out.bias").is_some());
        apply(&schema, tiny).unwrap().prepare().unwrap();
    }

    #[test]
    fn stable_diffusion_decoder_has_expected_shape() {
        let (_, output) = init(|cx| {
            let latent = cx.input(&[1, 8, 8, 4])?;
            autoencoder_kl_decoder(cx, &latent, &AutoencoderKlDecoderConfig::stable_diffusion())
        })
        .unwrap();
        assert_eq!(output.shape(), [1, 64, 64, 3]);
    }
}
