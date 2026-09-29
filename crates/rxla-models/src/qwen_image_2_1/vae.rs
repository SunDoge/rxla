//! Static-image specialization of `AutoencoderKLQwenImage21`.

use super::{Error, ModelDefinitionError, Result};
use rxla_core::{Conv2dOptions, DType, Tensor};
use rxla_nn::Cx;

const LATENT_CHANNELS: i64 = 64;
const OUTPUT_CHANNELS: i64 = 4;
const BASE_DIM: i64 = 144;
const DIM_MULTIPLIERS: [i64; 5] = [1, 2, 4, 8, 8];
const RESIDUAL_LAYERS: usize = 3;

fn convolution(kernel: i64) -> Conv2dOptions {
    Conv2dOptions {
        padding: [[kernel / 2, kernel / 2]; 2],
        ..Default::default()
    }
}

fn conv(cx: Cx, input: &Tensor, channels: i64, kernel: i64) -> Result<Tensor> {
    Ok(cx
        .conv2d(channels, [kernel, kernel])
        .options(convolution(kernel))
        .apply(input)?)
}

fn rms_norm(cx: Cx, input: &Tensor) -> Result<Tensor> {
    let channels = input.shape()[3];
    let gamma = cx
        .param("gamma", [channels, 1, 1, 1])?
        .reshape([channels])?;
    Ok(input.rms_norm(&gamma, 1e-12)?)
}

fn residual_block(cx: Cx, input: &Tensor, output_channels: i64) -> Result<Tensor> {
    let residual = if input.shape()[3] == output_channels {
        input.clone()
    } else {
        conv(cx.scope("conv_shortcut")?, input, output_channels, 1)?
    };
    let hidden = rms_norm(cx.scope("norm1")?, input)?.silu()?;
    let hidden = conv(cx.scope("conv1")?, &hidden, output_channels, 3)?;
    let hidden = rms_norm(cx.scope("norm2")?, &hidden)?.silu()?;
    Ok(conv(cx.scope("conv2")?, &hidden, output_channels, 3)?.add(&residual)?)
}

fn attention(cx: Cx, input: &Tensor) -> Result<Tensor> {
    let [batch, height, width, channels] = input.shape() else {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21VaeLatent,
        });
    };
    let hidden = rms_norm(cx.scope("norm")?, input)?;
    let qkv = conv(cx.scope("to_qkv")?, &hidden, channels * 3, 1)?
        .reshape([*batch, height * width, channels * 3])?
        .split(2, &[*channels, *channels, *channels])?;
    let query = qkv[0].unsqueeze(1)?;
    let key = qkv[1].unsqueeze(1)?;
    let value = qkv[2].unsqueeze(1)?;
    let attended = query
        .scaled_dot_product_attention(&key, &value, None, None)?
        .squeeze(1)?
        .reshape(input.shape())?;
    Ok(conv(cx.scope("proj")?, &attended, *channels, 1)?.add(input)?)
}

fn duplicate_upsample(input: &Tensor, output_channels: i64) -> Result<Tensor> {
    let [batch, height, width, input_channels] = input.shape() else {
        unreachable!("Qwen image VAE activations are NHWC")
    };
    let repeats = output_channels * 4 / input_channels;
    Ok(input
        .repeat_interleave(repeats, 3)?
        .reshape([*batch, *height, *width, output_channels, 2, 2])?
        .transpose([0, 1, 4, 2, 5, 3])?
        .reshape([*batch, height * 2, width * 2, output_channels])?)
}

fn declare_unused_temporal_conv(cx: Cx, channels: i64) -> Result<()> {
    // Diffusers declares this for video chunks. A static image is the first
    // chunk, where `QwenImage21Resample` deliberately bypasses it.
    let time = cx.scope("time_conv")?;
    time.param("weight", [channels * 2, channels, 1, 1])?;
    time.param("bias", [channels * 2])?;
    Ok(())
}

fn upsample(cx: Cx, input: &Tensor, channels: i64, temporal: bool) -> Result<Tensor> {
    if temporal {
        declare_unused_temporal_conv(cx.clone(), channels)?;
    }
    let resized = input.upsample_nearest2d([2, 2])?;
    conv(cx.scope("resample")?.scope("1")?, &resized, channels, 3)
}

fn residual_up_block(
    cx: Cx,
    input: &Tensor,
    output_channels: i64,
    upsample_stage: bool,
    temporal: bool,
) -> Result<Tensor> {
    let original = input.clone();
    let mut hidden = input.clone();
    let resnets = cx.scope("resnets")?;
    for index in 0..RESIDUAL_LAYERS {
        hidden = residual_block(resnets.scope(index.to_string())?, &hidden, output_channels)?;
    }
    if upsample_stage {
        hidden = upsample(cx.scope("upsampler")?, &hidden, output_channels, temporal)?;
        hidden = hidden.add(&duplicate_upsample(&original, output_channels)?)?;
    }
    Ok(hidden)
}

/// Decode `[batch, height * width, 64]` Qwen-Image-2.1 latents to an NHWC RGBA
/// image. The output is clamped to the checkpoint's `[-1, 1]` range.
pub fn qwen_image_2_1_vae_decoder(
    cx: Cx,
    tokens: &Tensor,
    latent_height: i64,
    latent_width: i64,
) -> Result<Tensor> {
    let [batch, sequence, channels] = tokens.shape() else {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21VaeLatent,
        });
    };
    if tokens.dtype() != DType::F32
        || latent_height <= 0
        || latent_width <= 0
        || *sequence != latent_height * latent_width
        || *channels != LATENT_CHANNELS
    {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21VaeLatent,
        });
    }
    let latent = tokens.reshape([*batch, latent_height, latent_width, LATENT_CHANNELS])?;
    let latent = conv(cx.scope("post_quant_conv")?, &latent, LATENT_CHANNELS, 1)?;
    let decoder = cx.scope("decoder")?;
    let deepest = BASE_DIM * DIM_MULTIPLIERS[DIM_MULTIPLIERS.len() - 1];
    let mut hidden = conv(decoder.scope("conv_in")?, &latent, deepest, 3)?;

    let middle = decoder.scope("mid_block")?;
    hidden = residual_block(middle.scope("resnets")?.scope("0")?, &hidden, deepest)?;
    hidden = attention(middle.scope("attentions")?.scope("0")?, &hidden)?;
    hidden = residual_block(middle.scope("resnets")?.scope("1")?, &hidden, deepest)?;

    let blocks = decoder.scope("up_blocks")?;
    for (stage, multiplier) in DIM_MULTIPLIERS.iter().rev().enumerate() {
        let output_channels = BASE_DIM * multiplier;
        hidden = residual_up_block(
            blocks.scope(stage.to_string())?,
            &hidden,
            output_channels,
            stage + 1 < DIM_MULTIPLIERS.len(),
            stage < 3,
        )?;
    }
    let hidden = rms_norm(decoder.scope("norm_out")?, &hidden)?.silu()?;
    Ok(conv(decoder.scope("conv_out")?, &hidden, OUTPUT_CHANNELS, 3)?.clamp(-1.0, 1.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::{Cx, Model};

    #[test]
    fn decoder_schema_matches_official_checkpoint_header() {
        let model = Model::new(|cx: Cx| {
            let latent = cx.input([1, 1, 64])?;
            qwen_image_2_1_vae_decoder(cx, &latent, 1, 1)
        })
        .trace()
        .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 16, 16, 4]);
        let parameters = model.schema().parameters();
        assert_eq!(parameters.len(), 134);
        assert_eq!(
            parameters
                .iter()
                .map(|parameter| parameter.shape().iter().product::<i64>())
                .sum::<i64>(),
            259_048_212
        );
        for path in [
            "post_quant_conv.weight",
            "decoder.mid_block.attentions.0.norm.gamma",
            "decoder.up_blocks.0.upsampler.time_conv.weight",
            "decoder.up_blocks.4.resnets.2.norm2.gamma",
            "decoder.norm_out.gamma",
            "decoder.conv_out.weight",
        ] {
            assert!(model.schema().get(path).is_some(), "missing {path}");
        }
        model.prepare().unwrap();
    }
}
