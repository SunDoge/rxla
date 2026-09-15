//! Functional conditional UNet topology.

use super::*;

/// Static architecture choices for a Diffusers-compatible conditional UNet.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    input_channels: i64,
    output_channels: i64,
    block_channels: Vec<i64>,
    layers_per_block: usize,
    down_cross_attention: Vec<bool>,
    up_cross_attention: Vec<bool>,
    context_width: i64,
    attention_heads: i64,
    norm_groups: i64,
    norm_epsilon: f32,
}

impl UnetConfig {
    pub fn tiny() -> Self {
        Self {
            input_channels: 4,
            output_channels: 4,
            block_channels: vec![32, 64],
            layers_per_block: 2,
            down_cross_attention: vec![false, true],
            up_cross_attention: vec![true, false],
            context_width: 32,
            attention_heads: 4,
            norm_groups: 32,
            norm_epsilon: 1e-5,
        }
    }

    pub fn stable_diffusion_v1() -> Self {
        Self {
            input_channels: 4,
            output_channels: 4,
            block_channels: vec![320, 640, 1_280, 1_280],
            layers_per_block: 2,
            down_cross_attention: vec![true, true, true, false],
            up_cross_attention: vec![false, true, true, true],
            context_width: 768,
            attention_heads: 8,
            norm_groups: 32,
            norm_epsilon: 1e-5,
        }
    }

    fn validate(&self) -> Result<()> {
        let stages = self.block_channels.len();
        if self.input_channels <= 0
            || self.output_channels <= 0
            || stages == 0
            || self.layers_per_block == 0
            || self.down_cross_attention.len() != stages
            || self.up_cross_attention.len() != stages
            || self.context_width <= 0
            || self.attention_heads <= 0
            || self.norm_groups <= 0
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || self.block_channels.iter().any(|&channels| {
                channels <= 0
                    || channels % self.norm_groups != 0
                    || channels % self.attention_heads != 0
            })
        {
            return Err(Error::InvalidDefinition {
                message: "invalid Stable Diffusion UNet configuration".into(),
            });
        }
        Ok(())
    }
}

fn convolution(padding: i64, stride: i64) -> Conv2dOptions {
    Conv2dOptions {
        strides: [stride; 2],
        padding: [[padding, padding]; 2],
        ..Default::default()
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

/// Conditional latent-diffusion UNet. Samples/results are NHWC; timestep input
/// is `[B, block_channels[0]]`; context is `[B, tokens, context_width]`.
pub fn unet(
    cx: &mut Cx,
    sample: &Tensor,
    timestep: &Tensor,
    context: &Tensor,
    config: &UnetConfig,
) -> Result<Tensor> {
    config.validate()?;
    let base = config.block_channels[0];
    if sample.shape().len() != 4
        || sample.shape()[3] != config.input_channels
        || timestep.shape() != [sample.shape()[0], base]
        || context.shape().len() != 3
        || context.shape()[0] != sample.shape()[0]
        || context.shape()[2] != config.context_width
    {
        return Err(Error::InvalidDefinition {
            message: "UNet input shapes do not match its configuration".into(),
        });
    }
    let time_width = base
        .checked_mul(4)
        .ok_or_else(|| Error::InvalidDefinition {
            message: "UNet timestep width overflow".into(),
        })?;
    let temb = cx.scope("time_embedding", |cx| {
        timestep_embedding(cx, timestep, time_width)
    })?;
    let mut hidden = cx
        .named("conv_in")?
        .conv2d(base, [3, 3])
        .options(convolution(1, 1))
        .apply(sample)?;
    let mut residuals = vec![hidden.clone()];

    for (stage, &out_channels) in config.block_channels.iter().enumerate() {
        hidden = indexed_scope(cx, "down_blocks", stage, |cx| {
            let mut value = hidden;
            for layer in 0..config.layers_per_block {
                value = indexed_scope(cx, "resnets", layer, |cx| {
                    resnet2d(
                        cx,
                        &value,
                        &temb,
                        Resnet2dOptions::new(out_channels)
                            .groups(config.norm_groups)
                            .epsilon(config.norm_epsilon),
                    )
                })?;
                if config.down_cross_attention[stage] {
                    value = indexed_scope(cx, "attentions", layer, |cx| {
                        spatial_transformer(
                            cx,
                            &value,
                            context,
                            SpatialTransformerOptions::new(out_channels / config.attention_heads)
                                .groups(config.norm_groups),
                        )
                    })?;
                }
                residuals.push(value.clone());
            }
            if stage + 1 < config.block_channels.len() {
                value = cx.scope("downsamplers", |cx| {
                    cx.scope("0", |cx| {
                        cx.named("conv")?
                            .conv2d(out_channels, [3, 3])
                            .options(convolution(1, 2))
                            .apply(&value)
                    })
                })?;
                residuals.push(value.clone());
            }
            Ok(value)
        })?;
    }

    hidden = cx.scope("mid_block", |cx| {
        let first = indexed_scope(cx, "resnets", 0, |cx| {
            resnet2d(
                cx,
                &hidden,
                &temb,
                Resnet2dOptions::new(hidden.shape()[3])
                    .groups(config.norm_groups)
                    .epsilon(config.norm_epsilon),
            )
        })?;
        let attended = indexed_scope(cx, "attentions", 0, |cx| {
            spatial_transformer(
                cx,
                &first,
                context,
                SpatialTransformerOptions::new(first.shape()[3] / config.attention_heads)
                    .groups(config.norm_groups),
            )
        })?;
        indexed_scope(cx, "resnets", 1, |cx| {
            resnet2d(
                cx,
                &attended,
                &temb,
                Resnet2dOptions::new(attended.shape()[3])
                    .groups(config.norm_groups)
                    .epsilon(config.norm_epsilon),
            )
        })
    })?;

    for up_stage in 0..config.block_channels.len() {
        let channel_stage = config.block_channels.len() - 1 - up_stage;
        let out_channels = config.block_channels[channel_stage];
        hidden = indexed_scope(cx, "up_blocks", up_stage, |cx| {
            let mut value = hidden;
            for layer in 0..=config.layers_per_block {
                let residual = residuals.pop().ok_or_else(|| Error::InvalidDefinition {
                    message: "UNet has too few down-block residuals".into(),
                })?;
                value = Tensor::concatenate(&[value, residual], 3)?;
                value = indexed_scope(cx, "resnets", layer, |cx| {
                    resnet2d(
                        cx,
                        &value,
                        &temb,
                        Resnet2dOptions::new(out_channels)
                            .groups(config.norm_groups)
                            .epsilon(config.norm_epsilon),
                    )
                })?;
                if config.up_cross_attention[up_stage] {
                    value = indexed_scope(cx, "attentions", layer, |cx| {
                        spatial_transformer(
                            cx,
                            &value,
                            context,
                            SpatialTransformerOptions::new(out_channels / config.attention_heads)
                                .groups(config.norm_groups),
                        )
                    })?;
                }
            }
            if up_stage + 1 < config.block_channels.len() {
                value = value.upsample_nearest2d([2, 2])?;
                value = cx.scope("upsamplers", |cx| {
                    cx.scope("0", |cx| {
                        cx.named("conv")?
                            .conv2d(out_channels, [3, 3])
                            .options(convolution(1, 1))
                            .apply(&value)
                    })
                })?;
            }
            Ok(value)
        })?;
    }
    if !residuals.is_empty() {
        return Err(Error::InvalidDefinition {
            message: "UNet forward left unused residuals".into(),
        });
    }
    let hidden = cx
        .named("conv_norm_out")?
        .group_norm(config.norm_groups)
        .epsilon(config.norm_epsilon)
        .apply(&hidden)?;
    cx.named("conv_out")?
        .conv2d(config.output_channels, [3, 3])
        .options(convolution(1, 1))
        .apply(&hidden.silu()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::{apply, init};

    fn tiny(cx: &mut Cx) -> Result<Tensor> {
        let sample = cx.input(&[1, 8, 8, 4])?;
        let timestep = cx.input(&[1, 32])?;
        let context = cx.input(&[1, 5, 32])?;
        unet(cx, &sample, &timestep, &context, &UnetConfig::tiny())
    }

    #[test]
    fn tiny_unet_builds_from_dataflow_shapes_and_lowers() {
        let (schema, output) = init(tiny).unwrap();
        assert_eq!(output.shape(), [1, 8, 8, 4]);
        for path in [
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.1.attentions.0.proj_in.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            "up_blocks.0.resnets.0.conv1.weight",
            "conv_norm_out.weight",
            "conv_out.bias",
        ] {
            assert!(schema.get(path).is_some(), "missing {path}");
        }
        apply(&schema, tiny).unwrap().prepare().unwrap();
    }
}
