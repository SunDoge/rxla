use super::{Flux2Config, Result, linear};
use rxla_core::{RotaryLayout, Tensor};
use rxla_nn::Cx;

pub(super) struct Modulation {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl Modulation {
    pub(super) fn new(cx: Cx, vector: &Tensor, hidden: i64) -> Result<Self> {
        let values = linear(cx.scope("lin")?, &vector.silu()?, hidden * 3)?
            .split(1, &[hidden, hidden, hidden])?;
        Ok(Self {
            shift: values[0].clone(),
            scale: values[1].clone(),
            gate: values[2].clone(),
        })
    }

    fn apply(&self, input: &Tensor, epsilon: f32) -> Result<Tensor> {
        let normalized = input.layer_norm(&[input.shape()[2]], None, None, epsilon)?;
        let shift = self.shift.unsqueeze(1)?.broadcast_to(input.shape())?;
        let scale = self.scale.unsqueeze(1)?.broadcast_to(input.shape())?;
        Ok(normalized.mul(&scale.add_scalar(1.0)?)?.add(&shift)?)
    }

    fn gated(&self, value: &Tensor) -> Result<Tensor> {
        Ok(value.mul(&self.gate.unsqueeze(1)?.broadcast_to(value.shape())?)?)
    }
}

pub(super) struct DoubleStreamModulation {
    attention: Modulation,
    mlp: Modulation,
}

impl DoubleStreamModulation {
    pub(super) fn new(cx: Cx, vector: &Tensor, hidden: i64) -> Result<Self> {
        let values = linear(cx.scope("lin")?, &vector.silu()?, hidden * 6)?
            .split(1, &[hidden, hidden, hidden, hidden, hidden, hidden])?;
        Ok(Self {
            attention: Modulation {
                shift: values[0].clone(),
                scale: values[1].clone(),
                gate: values[2].clone(),
            },
            mlp: Modulation {
                shift: values[3].clone(),
                scale: values[4].clone(),
                gate: values[5].clone(),
            },
        })
    }
}

fn qkv(cx: Cx, input: &Tensor, config: &Flux2Config) -> Result<(Tensor, Tensor, Tensor)> {
    let [batch, tokens, _] = input.shape() else {
        unreachable!("FLUX blocks use rank-three tensors")
    };
    let mixed = linear(cx.scope("qkv")?, input, config.hidden_size * 3)?.reshape(&[
        *batch,
        *tokens,
        3,
        config.num_heads,
        config.head_dim(),
    ])?;
    let query = mixed
        .narrow(2, 0, 1)?
        .squeeze(2)?
        .transpose(&[0, 2, 1, 3])?;
    let key = mixed
        .narrow(2, 1, 1)?
        .squeeze(2)?
        .transpose(&[0, 2, 1, 3])?;
    let value = mixed
        .narrow(2, 2, 1)?
        .squeeze(2)?
        .transpose(&[0, 2, 1, 3])?;
    let norm = cx.scope("norm")?;
    let query_scale = norm
        .scope("query_norm")?
        .param("scale", &[config.head_dim()])?;
    let key_scale = norm
        .scope("key_norm")?
        .param("scale", &[config.head_dim()])?;
    Ok((
        query.rms_norm(&query_scale, config.epsilon)?,
        key.rms_norm(&key_scale, config.epsilon)?,
        value,
    ))
}

fn attention_output(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    angles: &Tensor,
) -> Result<Tensor> {
    let query = query.rotary_embedding_angles(angles, RotaryLayout::Interleaved)?;
    let key = key.rotary_embedding_angles(angles, RotaryLayout::Interleaved)?;
    let attended = query.scaled_dot_product_attention(&key, value, None, None)?;
    Ok(attended.transpose(&[0, 2, 1, 3])?.reshape(&[
        attended.shape()[0],
        attended.shape()[2],
        attended.shape()[1] * attended.shape()[3],
    ])?)
}

fn swiglu(cx: Cx, input: &Tensor, hidden: i64, ratio: i64) -> Result<Tensor> {
    let width = hidden * ratio;
    let projected = linear(cx.scope("0")?, input, width * 2)?.split(2, &[width, width])?;
    let gated = projected[0].silu()?.mul(&projected[1])?;
    linear(cx.scope("2")?, &gated, hidden)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn double_block(
    cx: Cx,
    image: &Tensor,
    text: &Tensor,
    image_rope: &Tensor,
    text_rope: &Tensor,
    image_modulation: &DoubleStreamModulation,
    text_modulation: &DoubleStreamModulation,
    config: &Flux2Config,
) -> Result<(Tensor, Tensor)> {
    let image_attn = cx.scope("img_attn")?;
    let text_attn = cx.scope("txt_attn")?;
    let (image_q, image_k, image_v) = qkv(
        image_attn.clone(),
        &image_modulation.attention.apply(image, config.epsilon)?,
        config,
    )?;
    let (text_q, text_k, text_v) = qkv(
        text_attn.clone(),
        &text_modulation.attention.apply(text, config.epsilon)?,
        config,
    )?;
    let query = Tensor::concatenate(&[text_q, image_q], 2)?;
    let key = Tensor::concatenate(&[text_k, image_k], 2)?;
    let value = Tensor::concatenate(&[text_v, image_v], 2)?;
    let angles = Tensor::concatenate(&[text_rope.clone(), image_rope.clone()], 2)?;
    let attended = attention_output(&query, &key, &value, &angles)?;
    let text_tokens = text.shape()[1];
    let image_tokens = image.shape()[1];
    let text_delta = linear(
        text_attn.scope("proj")?,
        &attended.narrow(1, 0, text_tokens)?,
        config.hidden_size,
    )?;
    let image_delta = linear(
        image_attn.scope("proj")?,
        &attended.narrow(1, text_tokens, image_tokens)?,
        config.hidden_size,
    )?;
    let image = image.add(&image_modulation.attention.gated(&image_delta)?)?;
    let text = text.add(&text_modulation.attention.gated(&text_delta)?)?;
    let image_mlp = swiglu(
        cx.scope("img_mlp")?,
        &image_modulation.mlp.apply(&image, config.epsilon)?,
        config.hidden_size,
        config.mlp_ratio,
    )?;
    let text_mlp = swiglu(
        cx.scope("txt_mlp")?,
        &text_modulation.mlp.apply(&text, config.epsilon)?,
        config.hidden_size,
        config.mlp_ratio,
    )?;
    Ok((
        image.add(&image_modulation.mlp.gated(&image_mlp)?)?,
        text.add(&text_modulation.mlp.gated(&text_mlp)?)?,
    ))
}

pub(super) fn single_block(
    cx: Cx,
    input: &Tensor,
    angles: &Tensor,
    modulation: &Modulation,
    config: &Flux2Config,
) -> Result<Tensor> {
    let [batch, tokens, _] = input.shape() else {
        unreachable!("FLUX blocks use rank-three tensors")
    };
    let hidden = modulation.apply(input, config.epsilon)?;
    let mlp_width = config.hidden_size * config.mlp_ratio;
    let projected = linear(
        cx.scope("linear1")?,
        &hidden,
        config.hidden_size * 3 + mlp_width * 2,
    )?
    .split(2, &[config.hidden_size * 3, mlp_width * 2])?;
    let qkv = projected[0].reshape(&[*batch, *tokens, 3, config.num_heads, config.head_dim()])?;
    let query = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(&[0, 2, 1, 3])?;
    let key = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(&[0, 2, 1, 3])?;
    let value = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(&[0, 2, 1, 3])?;
    let norm = cx.scope("norm")?;
    let query_scale = norm
        .scope("query_norm")?
        .param("scale", &[config.head_dim()])?;
    let key_scale = norm
        .scope("key_norm")?
        .param("scale", &[config.head_dim()])?;
    let query = query.rms_norm(&query_scale, config.epsilon)?;
    let key = key.rms_norm(&key_scale, config.epsilon)?;
    let attended = attention_output(&query, &key, &value, angles)?;
    let mlp = projected[1].split(2, &[mlp_width, mlp_width])?;
    let mlp = mlp[0].silu()?.mul(&mlp[1])?;
    let mixed = Tensor::concatenate(&[attended, mlp], 2)?;
    let delta = linear(cx.scope("linear2")?, &mixed, config.hidden_size)?;
    Ok(input.add(&modulation.gated(&delta)?)?)
}
