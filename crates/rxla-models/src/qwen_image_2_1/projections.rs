//! Opt-in horizontal projection fusion, preserving checkpoint-native leaves.

use super::*;
use rxla_nn::Initializer;

pub(super) struct ProjectionGroup<'a, const N: usize> {
    names: [&'a str; N],
    width: i64,
    config: &'a QwenImage21Config,
}

impl<'a, const N: usize> ProjectionGroup<'a, N> {
    pub(super) fn new(names: [&'a str; N], width: i64, config: &'a QwenImage21Config) -> Self {
        Self {
            names,
            width,
            config,
        }
    }

    pub(super) fn apply(&self, cx: &Cx, input: &Tensor) -> Result<[Tensor; N]> {
        let in_features = input.shape()[input.shape().len() - 1];
        let mut weights = Vec::with_capacity(N);
        let mut scales = Vec::with_capacity(N);
        for name in self.names {
            let branch = cx.scope(name)?;
            if let Some(group) = self.config.quant_group_size {
                weights.push(branch.param_dtype("weight", [self.width, in_features], DType::U8)?);
                scales.push(branch.param("scale", [self.width, in_features / group])?);
            } else {
                weights.push(branch.param_initialized(
                    "weight",
                    [self.width, in_features],
                    Initializer::kaiming_uniform(),
                )?);
            }
        }
        let width = self.width * N as i64;
        let weight = Tensor::concatenate(&weights, 0)?;
        let weight = if let Some(group) = self.config.quant_group_size {
            let groups = in_features / group;
            let scale = Tensor::concatenate(&scales, 0)?
                .reshape([width, groups, 1])?
                .broadcast_to([width, groups, group])?;
            weight
                .cast(DType::F32)?
                .add_scalar(-128.0)?
                .reshape([width, groups, group])?
                .mul(&scale)?
                .reshape([width, in_features])?
        } else {
            weight
        };
        let output = if self.config.compute_dtype == DType::F32 {
            input.linear(&weight, None)?
        } else {
            input
                .cast(self.config.compute_dtype)?
                .linear(&weight.cast(self.config.compute_dtype)?, None)?
                .cast(DType::F32)?
        };
        let axis = output.shape().len() - 1;
        output
            .split(axis, &[self.width; N])?
            .try_into()
            .map_err(|_| Error::InvalidModel {
                kind: ModelDefinitionError::InvalidQwenImage21Configuration,
            })
    }
}
