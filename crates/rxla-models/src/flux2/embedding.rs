use super::{Flux2Config, Result, linear};
use rxla_core::Tensor;
use rxla_nn::Cx;

pub(super) fn timestep_embedding(
    cx: Cx,
    timestep: &Tensor,
    channels: i64,
    hidden: i64,
) -> Result<Tensor> {
    let half = channels / 2;
    let frequencies = (0..half)
        .map(|index| (-10_000.0_f32.ln() * index as f32 / half as f32).exp())
        .collect::<Vec<_>>();
    let frequency = cx.constant(&[1, half], &frequencies)?;
    let phase = timestep
        .mul_scalar(1000.0)?
        .unsqueeze(1)?
        .broadcast_to(&[timestep.shape()[0], half])?
        .mul(&frequency.broadcast_to(&[timestep.shape()[0], half])?)?;
    let embedding = Tensor::concatenate(&[phase.cos()?, phase.sin()?], 1)?;
    let hidden_value = linear(cx.scope("in_layer")?, &embedding, hidden)?.silu()?;
    linear(cx.scope("out_layer")?, &hidden_value, hidden)
}

pub(super) fn multi_axis_rope(cx: Cx, ids: &Tensor, config: &Flux2Config) -> Result<Tensor> {
    let [batch, tokens, _] = ids.shape() else {
        unreachable!("validated by flux2")
    };
    let mut axes = Vec::with_capacity(config.axes_dims_rope.len());
    for (axis, width) in config.axes_dims_rope.iter().copied().enumerate() {
        let pairs = width / 2;
        let frequencies = (0..pairs)
            .map(|index| config.rope_theta.powf(-2.0 * index as f32 / width as f32))
            .collect::<Vec<_>>();
        let frequency = cx.constant(&[1, 1, 1, pairs], &frequencies)?;
        let positions = ids.narrow(2, axis as i64, 1)?.unsqueeze(1)?;
        axes.push(
            positions
                .broadcast_to(&[*batch, 1, *tokens, pairs])?
                .mul(&frequency.broadcast_to(&[*batch, 1, *tokens, pairs])?)?,
        );
    }
    Ok(Tensor::concatenate(&axes, 3)?)
}
