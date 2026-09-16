use super::*;
use rxla_core::{Conv2dOptions, RotaryLayout};

fn projection(cx: &mut Cx, name: &str, input: &Tensor, width: i64, group: i64) -> Result<Tensor> {
    Ok(cx
        .scope(name)?
        .quantized_linear(width, group)
        .apply(input)?)
}

fn partial_rope(input: &Tensor, angles: &Tensor, width: i64) -> Result<Tensor> {
    let axis = input.shape().len() - 1;
    let rotated = input
        .narrow(axis, 0, width)?
        .rotary_embedding_angles(angles, RotaryLayout::SplitHalf)?;
    if width == input.shape()[axis] {
        return Ok(rotated);
    }
    Ok(Tensor::concatenate(
        &[
            rotated,
            input.narrow(axis, width, input.shape()[axis] - width)?,
        ],
        axis,
    )?)
}

pub(super) fn full_attention(
    cx: &mut Cx,
    input: &Tensor,
    angles: &Tensor,
    mask: &Tensor,
    config: &Qwen3_5Config,
) -> Result<Tensor> {
    let [batch, sequence, _] = input.shape() else {
        unreachable!("model validates rank")
    };
    let q = projection(
        cx,
        "q_proj",
        input,
        config.attention_heads * config.head_dim * 2,
        config.quant_group_size,
    )?
    .reshape(&[
        *batch,
        *sequence,
        config.attention_heads,
        config.head_dim * 2,
    ])?;
    let query = q.narrow(3, 0, config.head_dim)?;
    let gate = q
        .narrow(3, config.head_dim, config.head_dim)?
        .reshape(&[*batch, *sequence, config.attention_heads * config.head_dim])?
        .sigmoid()?;
    let query = cx
        .scope("q_norm")?
        .rms_norm()
        .epsilon(config.epsilon)
        .zero_centered(true)
        .apply(&query)?
        .transpose(&[0, 2, 1, 3])?;
    let key = projection(
        cx,
        "k_proj",
        input,
        config.key_value_heads * config.head_dim,
        config.quant_group_size,
    )?
    .reshape(&[*batch, *sequence, config.key_value_heads, config.head_dim])?;
    let key = cx
        .scope("k_norm")?
        .rms_norm()
        .epsilon(config.epsilon)
        .zero_centered(true)
        .apply(&key)?
        .transpose(&[0, 2, 1, 3])?;
    let value = projection(
        cx,
        "v_proj",
        input,
        config.key_value_heads * config.head_dim,
        config.quant_group_size,
    )?
    .reshape(&[*batch, *sequence, config.key_value_heads, config.head_dim])?
    .transpose(&[0, 2, 1, 3])?;
    let query = partial_rope(&query, angles, config.rotary_dim)?;
    let key = partial_rope(&key, angles, config.rotary_dim)?;
    let attended = query
        .grouped_query_attention(&key, &value, Some(mask), None)?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[*batch, *sequence, config.attention_heads * config.head_dim])?
        .mul(&gate)?;
    projection(
        cx,
        "o_proj",
        &attended,
        config.hidden_size,
        config.quant_group_size,
    )
}

fn l2_norm(input: &Tensor) -> Result<Tensor> {
    let axis = input.shape().len() - 1;
    let scale = input
        .square()?
        .sum(&[axis], true)?
        .add_scalar(1e-6)?
        .rsqrt()?
        .broadcast_to(input.shape())?;
    Ok(input.mul(&scale)?)
}

pub(super) fn gated_delta_attention(
    cx: &mut Cx,
    input: &Tensor,
    config: &Qwen3_5Config,
) -> Result<Tensor> {
    let [batch, sequence, _] = input.shape() else {
        unreachable!("model validates rank")
    };
    let width = config.linear_heads * config.linear_head_dim;
    let mixed = projection(cx, "in_proj_qkv", input, width * 3, config.quant_group_size)?
        .reshape(&[*batch, *sequence, 1, width * 3])?;
    let kernel = {
        let mut conv = cx.scope("conv1d")?;
        conv.param("weight", &[width * 3, 1, config.conv_kernel])?
            .reshape(&[width * 3, 1, config.conv_kernel, 1])?
    };
    let mixed = mixed
        .conv2d_oihw(
            &kernel,
            Conv2dOptions {
                padding: [[config.conv_kernel - 1, 0], [0, 0]],
                groups: width * 3,
                ..Default::default()
            },
        )?
        .reshape(&[*batch, *sequence, width * 3])?
        .silu()?;
    let query = l2_norm(&mixed.narrow(2, 0, width)?.reshape(&[
        *batch,
        *sequence,
        config.linear_heads,
        config.linear_head_dim,
    ])?)?
    .mul_scalar((config.linear_head_dim as f32).sqrt().recip())?;
    let key = l2_norm(&mixed.narrow(2, width, width)?.reshape(&[
        *batch,
        *sequence,
        config.linear_heads,
        config.linear_head_dim,
    ])?)?;
    let value = mixed.narrow(2, width * 2, width)?.reshape(&[
        *batch,
        *sequence,
        config.linear_heads,
        config.linear_head_dim,
    ])?;
    let z = projection(cx, "in_proj_z", input, width, config.quant_group_size)?.reshape(&[
        *batch,
        *sequence,
        config.linear_heads,
        config.linear_head_dim,
    ])?;
    let beta = projection(
        cx,
        "in_proj_b",
        input,
        config.linear_heads,
        config.quant_group_size,
    )?
    .sigmoid()?;
    let a = projection(
        cx,
        "in_proj_a",
        input,
        config.linear_heads,
        config.quant_group_size,
    )?;
    let a_log = cx.param("A_log", &[config.linear_heads])?;
    let dt_bias = cx.param("dt_bias", &[config.linear_heads])?;
    let decay = a.add(&dt_bias.broadcast_to(a.shape())?)?.softplus()?.mul(
        &a_log
            .exp()?
            .neg()?
            .reshape(&[1, 1, config.linear_heads])?
            .broadcast_to(a.shape())?,
    )?;
    let state_shape = [
        *batch,
        config.linear_heads,
        config.linear_head_dim,
        config.linear_head_dim,
    ];
    let mut state = cx.constant(&[], &[0.0])?.broadcast_to(&state_shape)?;
    let mut outputs = Vec::with_capacity(*sequence as usize);
    for position in 0..*sequence {
        let q = query.narrow(1, position, 1)?.squeeze(1)?;
        let k = key.narrow(1, position, 1)?.squeeze(1)?;
        let v = value.narrow(1, position, 1)?.squeeze(1)?;
        let beta = beta.narrow(1, position, 1)?.squeeze(1)?.unsqueeze(2)?;
        let decay = decay
            .narrow(1, position, 1)?
            .squeeze(1)?
            .exp()?
            .reshape(&[*batch, config.linear_heads, 1, 1])?
            .broadcast_to(&state_shape)?;
        state = state.mul(&decay)?;
        let predicted = state
            .mul(&k.unsqueeze(3)?.broadcast_to(&state_shape)?)?
            .sum(&[2], false)?;
        let delta = v.sub(&predicted)?.mul(&beta.broadcast_to(&[
            *batch,
            config.linear_heads,
            config.linear_head_dim,
        ])?)?;
        state = state.add(
            &k.unsqueeze(3)?
                .broadcast_to(&state_shape)?
                .mul(&delta.unsqueeze(2)?.broadcast_to(&state_shape)?)?,
        )?;
        outputs.push(
            state
                .mul(&q.unsqueeze(3)?.broadcast_to(&state_shape)?)?
                .sum(&[2], false)?
                .unsqueeze(1)?,
        );
    }
    let output = Tensor::concatenate(&outputs, 1)?;
    let axis = output.shape().len() - 1;
    let norm_scale = output
        .square()?
        .mean(&[axis], true)?
        .add_scalar(config.epsilon)?
        .rsqrt()?
        .broadcast_to(output.shape())?;
    let norm_weight = {
        let mut norm = cx.scope("norm")?;
        norm.param("weight", &[config.linear_head_dim])?
            .broadcast_to(output.shape())?
    };
    let output = output
        .mul(&norm_scale)?
        .mul(&norm_weight)?
        .mul(&z.silu()?)?
        .reshape(&[*batch, *sequence, width])?;
    projection(
        cx,
        "out_proj",
        &output,
        config.hidden_size,
        config.quant_group_size,
    )
}

pub(super) fn mlp(cx: &mut Cx, input: &Tensor, config: &Qwen3_5Config) -> Result<Tensor> {
    let gate = projection(
        cx,
        "gate_proj",
        input,
        config.intermediate_size,
        config.quant_group_size,
    )?
    .silu()?;
    let up = projection(
        cx,
        "up_proj",
        input,
        config.intermediate_size,
        config.quant_group_size,
    )?;
    projection(
        cx,
        "down_proj",
        &gate.mul(&up)?,
        config.hidden_size,
        config.quant_group_size,
    )
}
