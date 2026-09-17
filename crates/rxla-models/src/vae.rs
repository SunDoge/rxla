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
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidVaeConfiguration,
            });
        }
        Ok(())
    }
}

fn indexed_scope(cx: &Cx, collection: &str, index: usize) -> Result<Cx> {
    Ok(cx.scope_path([collection.to_owned(), index.to_string()])?)
}

fn resnet(
    cx: Cx,
    input: &Tensor,
    output_channels: i64,
    groups: i64,
    epsilon: f32,
) -> Result<Tensor> {
    let normalized = cx
        .scope("norm1")?
        .group_norm(groups)
        .epsilon(epsilon)
        .apply(input)?;
    let hidden = cx
        .scope("conv1")?
        .conv2d(output_channels, [3, 3])
        .options(convolution(3))
        .apply(&normalized.silu()?)?;
    let normalized = cx
        .scope("norm2")?
        .group_norm(groups)
        .epsilon(epsilon)
        .apply(&hidden)?;
    let hidden = cx
        .scope("conv2")?
        .conv2d(output_channels, [3, 3])
        .options(convolution(3))
        .apply(&normalized.silu()?)?;
    let residual = if input.shape()[3] == output_channels {
        input.clone()
    } else {
        cx.scope("conv_shortcut")?
            .conv2d(output_channels, [1, 1])
            .apply(input)?
    };
    Ok(hidden.add(&residual)?)
}

fn attention(cx: Cx, input: &Tensor, groups: i64, epsilon: f32) -> Result<Tensor> {
    let [batch, height, width, channels] = input.shape() else {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidVaeAttentionInput,
        });
    };
    let hidden = cx
        .scope("group_norm")?
        .group_norm(groups)
        .epsilon(epsilon)
        .apply(input)?
        .reshape(&[*batch, height * width, *channels])?;
    let q = hidden.apply(&cx.layer("to_q", Linear::new(*channels))?)?;
    let k = cx
        .scope("to_k")?
        .linear(*channels)
        .apply(&hidden)?
        .transpose(&[0, 2, 1])?;
    let v = hidden.apply(&cx.layer("to_v", Linear::new(*channels))?)?;
    let attended = q
        .matmul(&k)?
        .mul_scalar(1.0 / (*channels as f32).sqrt())?
        .softmax(2)?
        .matmul(&v)?;
    let to_out = cx.scope("to_out")?;
    Ok(to_out
        .scope("0")?
        .linear(*channels)
        .apply(&attended)?
        .reshape(input.shape())?
        .add(input)?)
}

/// Decode an NHWC latent into an NHWC image while interpreting parameter
/// declarations from the surrounding [`Cx`].
pub fn autoencoder_kl_decoder(
    cx: Cx,
    latent: &Tensor,
    config: &AutoencoderKlDecoderConfig,
) -> Result<Tensor> {
    config.validate()?;
    if latent.shape().len() != 4 || latent.shape()[3] != config.latent_channels {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidVaeLatentShape,
        });
    }
    let deepest = *config.block_channels.last().ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::MissingVaeBlockChannels,
    })?;
    let mut hidden = latent.mul_scalar(1.0 / config.scaling_factor)?;
    hidden = cx
        .scope("post_quant_conv")?
        .conv2d(config.latent_channels, [1, 1])
        .apply(&hidden)?;
    let decoder = cx.scope("decoder")?;
    hidden = decoder
        .scope("conv_in")?
        .conv2d(deepest, [3, 3])
        .options(convolution(3))
        .apply(&hidden)?;
    hidden = {
        let mid = decoder.scope("mid_block")?;
        let first = {
            let scope = indexed_scope(&mid, "resnets", 0)?;
            resnet(
                scope,
                &hidden,
                deepest,
                config.norm_groups,
                config.norm_epsilon,
            )?
        };
        let attended = {
            let scope = indexed_scope(&mid, "attentions", 0)?;
            attention(scope, &first, config.norm_groups, config.norm_epsilon)?
        };
        let scope = indexed_scope(&mid, "resnets", 1)?;
        resnet(
            scope,
            &attended,
            deepest,
            config.norm_groups,
            config.norm_epsilon,
        )?
    };

    let stages = config.block_channels.len();
    for (stage, &output_channels) in config.block_channels.iter().rev().enumerate() {
        let block = indexed_scope(&decoder, "up_blocks", stage)?;
        let mut value = hidden;
        for layer in 0..=config.layers_per_block {
            let scope = indexed_scope(&block, "resnets", layer)?;
            value = resnet(
                scope,
                &value,
                output_channels,
                config.norm_groups,
                config.norm_epsilon,
            )?;
        }
        if stage + 1 < stages {
            let scope = block.scope_path(["upsamplers", "0"])?;
            value = scope
                .scope("conv")?
                .conv2d(output_channels, [3, 3])
                .options(convolution(3))
                .apply(&value.upsample_nearest2d([2, 2])?)?;
        }
        hidden = value;
    }
    let normalized = decoder
        .scope("conv_norm_out")?
        .group_norm(config.norm_groups)
        .epsilon(config.norm_epsilon)
        .apply(&hidden)?;
    Ok(decoder
        .scope("conv_out")?
        .conv2d(config.output_channels, [3, 3])
        .options(convolution(3))
        .apply(&normalized.silu()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::Model;

    fn tiny(cx: Cx) -> Result<Tensor> {
        let latent = cx.input(&[1, 8, 6, 4])?;
        autoencoder_kl_decoder(cx, &latent, &AutoencoderKlDecoderConfig::tiny())
    }

    #[test]
    fn tiny_decoder_has_stable_schema_and_lowers() {
        let model = Model::new(tiny).trace().unwrap();
        let schema = model.schema();
        assert_eq!(model.outputs()[0].shape(), [1, 16, 12, 3]);
        assert_eq!(schema.parameters().len(), 70);
        assert_eq!(
            schema.get("decoder.conv_in.weight").unwrap().shape(),
            [64, 4, 3, 3]
        );
        assert!(schema.get("decoder.conv_out.bias").is_some());
        model.prepare().unwrap();
    }

    #[test]
    fn stable_diffusion_decoder_has_expected_shape() {
        let model = Model::new(|cx: Cx| {
            let latent = cx.input(&[1, 8, 8, 4])?;
            autoencoder_kl_decoder(cx, &latent, &AutoencoderKlDecoderConfig::stable_diffusion())
        })
        .trace()
        .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 64, 64, 3]);
    }
}
