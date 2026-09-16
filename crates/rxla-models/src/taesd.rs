//! Tiny AutoEncoder decoder used by Stable Diffusion TAESD checkpoints.

use rxla_core::{Conv2dOptions, Tensor};
use rxla_nn::{Cx, Error};

use crate::Result;

fn convolution() -> Conv2dOptions {
    Conv2dOptions {
        padding: [[1, 1], [1, 1]],
        ..Default::default()
    }
}

fn tiny_block(cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
    let mut conv = cx.scope("conv")?;
    let hidden = conv
        .layer("0")?
        .conv2d(input.shape()[3], [3, 3])
        .options(convolution())
        .apply(input)?
        .relu()?;
    let hidden = conv
        .layer("2")?
        .conv2d(input.shape()[3], [3, 3])
        .options(convolution())
        .apply(&hidden)?
        .relu()?;
    let hidden = conv
        .layer("4")?
        .conv2d(input.shape()[3], [3, 3])
        .options(convolution())
        .apply(&hidden)?;
    Ok(hidden.add(input)?.relu()?)
}

fn transition(
    cx: &mut Cx,
    layer: usize,
    input: &Tensor,
    output_channels: i64,
    bias: bool,
) -> Result<Tensor> {
    cx.layer(&layer.to_string())?
        .conv2d(output_channels, [3, 3])
        .options(convolution())
        .bias(bias)
        .apply(input)
}

/// Diffusers-compatible `AutoencoderTiny.decoder` using NHWC tensors.
///
/// The input width is validated where it is used. Parameter paths match the
/// PyTorch sequential indices in TAESD safetensors checkpoints. Three nearest
/// neighbor stages produce RGB output at eight times the spatial resolution.
pub fn taesd_decoder(cx: &mut Cx, input: &Tensor) -> Result<Tensor> {
    if input.shape().len() != 4 || input.shape()[3] != 4 {
        return Err(Error::InvalidDefinition {
            message: "TAESD decoder expects NHWC input with four latent channels".into(),
        });
    }

    let mut decoder = cx.scope("decoder")?;
    let mut layers = decoder.scope("layers")?;
    let mut hidden = input.mul_scalar(1. / 3.)?.tanh()?.mul_scalar(3.)?;
    hidden = transition(&mut layers, 0, &hidden, 64, true)?.relu()?;

    let mut next_layer = 2;
    for (stage_index, block_count) in [3, 3, 3, 1].into_iter().enumerate() {
        for _ in 0..block_count {
            let mut block = layers.scope(&next_layer.to_string())?;
            hidden = tiny_block(&mut block, &hidden)?;
            next_layer += 1;
        }
        let upsample = stage_index != 3;
        if upsample {
            hidden = hidden.upsample_nearest2d([2, 2])?;
            next_layer += 1;
        }
        let output_channels = if stage_index == 3 { 3 } else { 64 };
        hidden = transition(
            &mut layers,
            next_layer,
            &hidden,
            output_channels,
            stage_index == 3,
        )?;
        next_layer += 1;
    }
    debug_assert_eq!(next_layer, 19);
    Ok(hidden.mul_scalar(2.)?.add_scalar(-1.)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::{apply, init};

    fn tiny(cx: &mut Cx) -> Result<Tensor> {
        let latent = cx.input(&[1, 8, 12, 4])?;
        taesd_decoder(cx, &latent)
    }

    #[test]
    fn canonical_decoder_has_checkpoint_schema_and_lowers() {
        let (schema, output) = init(tiny).unwrap();
        assert_eq!(output.shape(), [1, 64, 96, 3]);
        assert_eq!(schema.parameters().len(), 67);
        for name in [
            "decoder.layers.0.weight",
            "decoder.layers.0.bias",
            "decoder.layers.2.conv.0.weight",
            "decoder.layers.2.conv.4.bias",
            "decoder.layers.17.conv.4.weight",
            "decoder.layers.18.weight",
            "decoder.layers.18.bias",
        ] {
            assert!(schema.get(name).is_some(), "missing {name}");
        }
        assert!(schema.get("decoder.layers.6.bias").is_none());
        apply(&schema, tiny).unwrap().prepare().unwrap();
    }

    #[test]
    fn rejects_non_latent_channels() {
        let error = init(|cx| {
            let image = cx.input(&[1, 8, 8, 3])?;
            taesd_decoder(cx, &image)
        });
        assert!(error.is_err());
    }
}
