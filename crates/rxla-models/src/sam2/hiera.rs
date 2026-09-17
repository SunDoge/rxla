use super::HieraConfig;
use crate::{Error, ModelDefinitionError, Result};
use derive_setters::Setters;
use rxla_core::{Conv2dOptions, DType, Pool2dOptions, Tensor};
use rxla_nn::{Conv2d, Cx, LayerNorm, Linear, TensorApply, path};

fn window_partition(input: &Tensor, window: i64) -> Result<(Tensor, [i64; 2])> {
    let [batch, height, width, channels] =
        <[i64; 4]>::try_from(input.shape()).map_err(|_| Error::InvalidModel {
            kind: ModelDefinitionError::InvalidSam2Image,
        })?;
    let pad_height = (window - height % window) % window;
    let pad_width = (window - width % window) % window;
    let padded = input.pad(&[[0, 0], [0, pad_height], [0, pad_width], [0, 0]], 0.0)?;
    let padded_height = height + pad_height;
    let padded_width = width + pad_width;
    let windows = padded
        .reshape(&[
            batch,
            padded_height / window,
            window,
            padded_width / window,
            window,
            channels,
        ])?
        .transpose(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[
            batch * (padded_height / window) * (padded_width / window),
            window,
            window,
            channels,
        ])?;
    Ok((windows, [padded_height, padded_width]))
}

fn window_unpartition(
    windows: &Tensor,
    window: i64,
    padded: [i64; 2],
    original: [i64; 2],
) -> Result<Tensor> {
    let [padded_height, padded_width] = padded;
    let windows_per_image = padded_height / window * (padded_width / window);
    let batch = windows.shape()[0] / windows_per_image;
    let channels = windows.shape()[3];
    let output = windows
        .reshape(&[
            batch,
            padded_height / window,
            padded_width / window,
            window,
            window,
            channels,
        ])?
        .transpose(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[batch, padded_height, padded_width, channels])?;
    Ok(output
        .narrow(1, 0, original[0])?
        .narrow(2, 0, original[1])?)
}

fn pool(input: &Tensor, stride: [i64; 2]) -> Result<Tensor> {
    Ok(input.max_pool2d(Pool2dOptions {
        window: stride,
        strides: stride,
        padding: [[0, 0], [0, 0]],
    })?)
}

#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
struct MultiScaleAttention {
    dim_out: i64,
    heads: i64,
    #[setters(generate)]
    q_stride: Option<[i64; 2]>,
}

impl MultiScaleAttention {
    fn new(dim_out: i64, heads: i64) -> Self {
        Self {
            dim_out,
            heads,
            q_stride: None,
        }
    }

    fn call(&self, cx: Cx, input: &Tensor) -> Result<Tensor> {
        let [batch, height, width, _] = <[i64; 4]>::try_from(input.shape()).unwrap();
        let head_dim = self.dim_out / self.heads;
        let qkv = input
            .through(path!(cx / "qkv")?, Linear::new(self.dim_out * 3))?
            .reshape(&[batch, height * width, 3, self.heads, head_dim])?
            .unbind(2)?;
        let [mut query, key, value] =
            <[Tensor; 3]>::try_from(qkv).map_err(|_| Error::InvalidModel {
                kind: ModelDefinitionError::InvalidSam2HieraConfiguration,
            })?;
        let (query_height, query_width) = if let Some(stride) = self.q_stride {
            query = pool(
                &query.reshape(&[batch, height, width, self.dim_out])?,
                stride,
            )?
            .reshape(&[
                batch,
                height / stride[0] * (width / stride[1]),
                self.heads,
                head_dim,
            ])?;
            (height / stride[0], width / stride[1])
        } else {
            (height, width)
        };
        let query = query.transpose(&[0, 2, 1, 3])?;
        let key = key.transpose(&[0, 2, 1, 3])?;
        let value = value.transpose(&[0, 2, 1, 3])?;
        let output = query
            .scaled_dot_product_attention(&key, &value, None, None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[batch, query_height, query_width, self.dim_out])?;
        Ok(output.through(path!(cx / "proj")?, Linear::new(self.dim_out))?)
    }
}

#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
struct MultiScaleBlock {
    dim: i64,
    dim_out: i64,
    heads: i64,
    #[setters(generate)]
    window_size: i64,
    #[setters(generate)]
    q_stride: Option<[i64; 2]>,
}

impl MultiScaleBlock {
    fn new(dim: i64, dim_out: i64, heads: i64) -> Self {
        Self {
            dim,
            dim_out,
            heads,
            window_size: 0,
            q_stride: None,
        }
    }

    fn call(&self, cx: Cx, input: &Tensor) -> Result<Tensor> {
        let normalized = input.through(path!(cx / "norm1")?, LayerNorm::new(1).epsilon(1e-6))?;
        let shortcut = if self.dim == self.dim_out {
            input.clone()
        } else {
            let projected = normalized.through(path!(cx / "proj")?, Linear::new(self.dim_out))?;
            pool(
                &projected,
                self.q_stride
                    .expect("dimension changes only at pooled blocks"),
            )?
        };

        let [height, width] = [normalized.shape()[1], normalized.shape()[2]];
        let (partitioned, padded) = if self.window_size > 0 {
            window_partition(&normalized, self.window_size)?
        } else {
            (normalized, [height, width])
        };
        let attention = MultiScaleAttention::new(self.dim_out, self.heads).q_stride(self.q_stride);
        let mut hidden = attention.call(path!(cx / "attn")?, &partitioned)?;
        if self.window_size > 0 {
            let output_window = if let Some(stride) = self.q_stride {
                self.window_size / stride[0]
            } else {
                self.window_size
            };
            let output_padded = if self.q_stride.is_some() {
                let height = shortcut.shape()[1];
                let width = shortcut.shape()[2];
                [
                    height + (output_window - height % output_window) % output_window,
                    width + (output_window - width % output_window) % output_window,
                ]
            } else {
                padded
            };
            hidden = window_unpartition(
                &hidden,
                output_window,
                output_padded,
                [shortcut.shape()[1], shortcut.shape()[2]],
            )?;
        }
        hidden = shortcut.add(&hidden)?;
        let mlp = hidden
            .through(path!(cx / "norm2")?, LayerNorm::new(1).epsilon(1e-6))?
            .through(
                path!(cx / "mlp" / "layers" / 0)?,
                Linear::new(self.dim_out * 4),
            )?
            .gelu()?
            .through(path!(cx / "mlp" / "layers" / 1)?, Linear::new(self.dim_out))?;
        Ok(hidden.add(&mlp)?)
    }
}

fn position_embedding(cx: Cx, shape: [i64; 2], config: &HieraConfig) -> Result<Tensor> {
    let base = cx
        .param(
            "pos_embed",
            &[
                1,
                config.embed_dim,
                config.position_embedding_size[0],
                config.position_embedding_size[1],
            ],
        )?
        .transpose(&[0, 2, 3, 1])?
        .resize_bicubic2d(shape)?;
    let window = config.window_spec[0];
    let tiled = cx
        .param("pos_embed_window", &[1, config.embed_dim, window, window])?
        .transpose(&[0, 2, 3, 1])?
        .reshape(&[1, 1, window, 1, window, config.embed_dim])?
        .broadcast_to(&[
            1,
            shape[0] / window,
            window,
            shape[1] / window,
            window,
            config.embed_dim,
        ])?
        .reshape(&[1, shape[0], shape[1], config.embed_dim])?;
    Ok(base.add(&tiled)?)
}

/// Apply the native SAM 2 Hiera image trunk to an NHWC F32 image.
///
/// Returned stage features use the checkpoint-native NCHW layout and are
/// ordered from highest to lowest spatial resolution.
pub fn hiera(cx: Cx, image: &Tensor, config: &HieraConfig) -> Result<[Tensor; 4]> {
    config.validate()?;
    let cx = cx.with_parameter_storage_dtype(config.parameter_dtype)?;
    if image.dtype() != DType::F32
        || image.shape().len() != 4
        || image.shape().iter().any(|&dimension| dimension <= 0)
        || image.shape()[3] != 3
        || image.shape()[1] % 32 != 0
        || image.shape()[2] % 32 != 0
    {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidSam2Image,
        });
    }
    let mut hidden = image.through(
        path!(cx / "patch_embed" / "proj")?,
        Conv2d::new(config.embed_dim, [7, 7]).options(Conv2dOptions {
            strides: [4, 4],
            padding: [[3, 3], [3, 3]],
            ..Default::default()
        }),
    )?;
    hidden = hidden.add(&position_embedding(
        cx.clone(),
        [hidden.shape()[1], hidden.shape()[2]],
        config,
    )?)?;

    let stage_ends = config
        .stages
        .iter()
        .scan(0usize, |sum, &depth| {
            *sum += depth;
            Some(*sum - 1)
        })
        .collect::<Vec<_>>();
    let q_pool_blocks = stage_ends
        .iter()
        .take(config.q_pool)
        .map(|index| index + 1)
        .collect::<Vec<_>>();
    let mut outputs = Vec::with_capacity(config.stages.len());
    let mut dim = config.embed_dim;
    let mut heads = config.num_heads;
    let mut stage = 0;
    for index in 0..config.stages.iter().sum() {
        let window_stage = stage;
        let transition = index > 0 && stage_ends.contains(&(index - 1));
        let dim_out = if transition { dim * 2 } else { dim };
        if transition {
            heads *= 2;
            stage += 1;
        }
        let window = if config.global_attention_blocks.contains(&index) {
            0
        } else {
            config.window_spec[window_stage]
        };
        let block_name = index;
        let block = MultiScaleBlock::new(dim, dim_out, heads)
            .window_size(window)
            .q_stride(q_pool_blocks.contains(&index).then_some(config.q_stride));
        hidden = block.call(path!(cx / "blocks" / block_name)?, &hidden)?;
        dim = dim_out;
        if stage_ends.contains(&index) {
            outputs.push(hidden.transpose(&[0, 3, 1, 2])?);
        }
    }
    <[Tensor; 4]>::try_from(outputs).map_err(|_| Error::InvalidModel {
        kind: ModelDefinitionError::InvalidSam2HieraConfiguration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_nn::{Model, ModelInput};

    fn small_trunk(cx: Cx, image: Tensor) -> crate::Result<[Tensor; 4]> {
        hiera(
            path!(cx / "image_encoder" / "trunk")?,
            &image,
            &HieraConfig::sam2_1_small(),
        )
    }

    #[test]
    fn sam2_1_small_hiera_has_official_stage_shapes_and_paths() {
        let model = Model::new(small_trunk)
            .inputs(ModelInput::new([1, 1024, 1024, 3]))
            .trace()
            .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 96, 256, 256]);
        assert_eq!(model.outputs()[1].shape(), [1, 192, 128, 128]);
        assert_eq!(model.outputs()[2].shape(), [1, 384, 64, 64]);
        assert_eq!(model.outputs()[3].shape(), [1, 768, 32, 32]);
        assert!(
            model.schema().parameters().iter().any(
                |parameter| parameter.path() == "image_encoder.trunk.blocks.14.attn.qkv.weight"
            )
        );
    }
}
