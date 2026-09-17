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
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidUnetConfiguration,
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

fn indexed_scope(cx: &Cx, collection: &str, index: usize) -> Result<Cx> {
    Ok(cx.scope_path([collection.to_owned(), index.to_string()])?)
}

/// Conditional latent-diffusion UNet. Samples/results are NHWC; timestep input
/// is `[B, block_channels[0]]`; context is `[B, tokens, context_width]`.
pub fn unet(
    cx: Cx,
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
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidUnetInput,
        });
    }
    let time_width = base.checked_mul(4).ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::UnetTimestepWidthOverflow,
    })?;
    let temb = {
        let scope = cx.scope("time_embedding")?;
        timestep_embedding(scope, timestep, time_width)?
    };
    let mut hidden = cx
        .scope("conv_in")?
        .conv2d(base, [3, 3])
        .options(convolution(1, 1))
        .apply(sample)?;
    let mut residuals = vec![hidden.clone()];

    for (stage, &out_channels) in config.block_channels.iter().enumerate() {
        let block = indexed_scope(&cx, "down_blocks", stage)?;
        let mut value = hidden;
        for layer in 0..config.layers_per_block {
            {
                let scope = indexed_scope(&block, "resnets", layer)?;
                value = resnet2d(
                    scope,
                    &value,
                    &temb,
                    Resnet2dOptions::new(out_channels)
                        .groups(config.norm_groups)
                        .epsilon(config.norm_epsilon),
                )?;
            }
            if config.down_cross_attention[stage] {
                let scope = indexed_scope(&block, "attentions", layer)?;
                value = spatial_transformer(
                    scope,
                    &value,
                    context,
                    SpatialTransformerOptions::new(out_channels / config.attention_heads)
                        .groups(config.norm_groups),
                )?;
            }
            residuals.push(value.clone());
        }
        if stage + 1 < config.block_channels.len() {
            let scope = block.scope_path(["downsamplers", "0"])?;
            value = scope
                .scope("conv")?
                .conv2d(out_channels, [3, 3])
                .options(convolution(1, 2))
                .apply(&value)?;
            residuals.push(value.clone());
        }
        hidden = value;
    }

    hidden = {
        let mid = cx.scope("mid_block")?;
        let first = {
            let scope = indexed_scope(&mid, "resnets", 0)?;
            resnet2d(
                scope,
                &hidden,
                &temb,
                Resnet2dOptions::new(hidden.shape()[3])
                    .groups(config.norm_groups)
                    .epsilon(config.norm_epsilon),
            )?
        };
        let attended = {
            let scope = indexed_scope(&mid, "attentions", 0)?;
            spatial_transformer(
                scope,
                &first,
                context,
                SpatialTransformerOptions::new(first.shape()[3] / config.attention_heads)
                    .groups(config.norm_groups),
            )?
        };
        let scope = indexed_scope(&mid, "resnets", 1)?;
        resnet2d(
            scope,
            &attended,
            &temb,
            Resnet2dOptions::new(attended.shape()[3])
                .groups(config.norm_groups)
                .epsilon(config.norm_epsilon),
        )?
    };

    for up_stage in 0..config.block_channels.len() {
        let channel_stage = config.block_channels.len() - 1 - up_stage;
        let out_channels = config.block_channels[channel_stage];
        let block = indexed_scope(&cx, "up_blocks", up_stage)?;
        let mut value = hidden;
        for layer in 0..=config.layers_per_block {
            let residual = residuals.pop().ok_or(Error::InvalidModel {
                kind: ModelDefinitionError::MissingUnetResidual,
            })?;
            value = Tensor::concatenate(&[value, residual], 3)?;
            {
                let scope = indexed_scope(&block, "resnets", layer)?;
                value = resnet2d(
                    scope,
                    &value,
                    &temb,
                    Resnet2dOptions::new(out_channels)
                        .groups(config.norm_groups)
                        .epsilon(config.norm_epsilon),
                )?;
            }
            if config.up_cross_attention[up_stage] {
                let scope = indexed_scope(&block, "attentions", layer)?;
                value = spatial_transformer(
                    scope,
                    &value,
                    context,
                    SpatialTransformerOptions::new(out_channels / config.attention_heads)
                        .groups(config.norm_groups),
                )?;
            }
        }
        if up_stage + 1 < config.block_channels.len() {
            value = value.upsample_nearest2d([2, 2])?;
            let scope = block.scope_path(["upsamplers", "0"])?;
            value = scope
                .scope("conv")?
                .conv2d(out_channels, [3, 3])
                .options(convolution(1, 1))
                .apply(&value)?;
        }
        hidden = value;
    }
    if !residuals.is_empty() {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::UnusedUnetResidual,
        });
    }
    let hidden = cx
        .scope("conv_norm_out")?
        .group_norm(config.norm_groups)
        .epsilon(config.norm_epsilon)
        .apply(&hidden)?;
    Ok(cx
        .scope("conv_out")?
        .conv2d(config.output_channels, [3, 3])
        .options(convolution(1, 1))
        .apply(&hidden.silu()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::Model;

    fn tiny(cx: Cx) -> Result<Tensor> {
        let sample = cx.input(&[1, 8, 8, 4])?;
        let timestep = cx.input(&[1, 32])?;
        let context = cx.input(&[1, 5, 32])?;
        unet(cx, &sample, &timestep, &context, &UnetConfig::tiny())
    }

    #[test]
    fn tiny_unet_builds_from_dataflow_shapes_and_lowers() {
        let model = Model::new(tiny).trace().unwrap();
        let schema = model.schema();
        assert_eq!(model.outputs()[0].shape(), [1, 8, 8, 4]);
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
        model.prepare().unwrap();
    }
}
