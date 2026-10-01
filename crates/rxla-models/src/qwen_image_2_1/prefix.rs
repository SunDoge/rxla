//! Causal-prefix factoring. No approximation or timestep-dependent text cache.

use super::*;

pub(super) fn blocks(
    cx: &Cx,
    image: &Tensor,
    context: &Tensor,
    modulation: &Tensor,
    angles: &Tensor,
    text_bias: &Tensor,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    let batch = image.shape()[0];
    let text_tokens = context.shape()[1];
    let image_tokens = image.shape()[1];
    // Compute zero-time modulation independently, not by slicing a computation
    // that also consumes the current timestep. Parameter declarations reuse
    // the exact same named leaves as the image branch.
    let zero = cx.constant([1], [0.0])?;
    let zero_time = timestep_embedding(
        cx.scope("time_text_embed")?,
        &zero,
        config.hidden_size,
        config,
    )?;
    let text_modulation = linear(
        cx.scope("modulation")?.scope("1")?,
        &zero_time.silu()?,
        config.hidden_size * 4,
        false,
        config,
    )?
    .reshape([1, 1, config.hidden_size * 4])?
    .broadcast_to([batch, text_tokens, config.hidden_size * 4])?;
    let image_modulation = modulation.narrow(1, text_tokens, image_tokens)?;
    let text_angles = angles.narrow(2, 0, text_tokens)?;
    let image_angles = angles.narrow(2, text_tokens, image_tokens)?;
    let blocks = cx.scope("transformer_blocks")?;
    let mut text = context.clone();
    let mut image = image.clone();
    for index in 0..config.num_layers {
        let block = blocks.scope(index.to_string())?;
        let mut prefix = None;
        text = transformer_block(
            block.clone(),
            &text,
            &text_modulation,
            &text_angles,
            text_bias,
            AttentionRegion::Text(&mut prefix),
            config,
        )?;
        let (key, value) = prefix.ok_or(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21Configuration,
        })?;
        image = transformer_block(
            block,
            &image,
            &image_modulation,
            &image_angles,
            text_bias,
            AttentionRegion::Image {
                key: &key,
                value: &value,
            },
            config,
        )?;
    }
    Ok(image)
}
