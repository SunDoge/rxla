use crate::{Error, ModelDefinitionError, Result};
use derive_setters::Setters;
use rxla_core::DType;

/// Checkpoint-defined Hiera trunk structure used by SAM 2's image encoder.
#[derive(Clone, Debug, PartialEq, Eq, Setters)]
#[setters(generate = false)]
pub struct HieraConfig {
    pub embed_dim: i64,
    pub num_heads: i64,
    pub stages: Vec<usize>,
    pub q_pool: usize,
    pub q_stride: [i64; 2],
    pub window_spec: Vec<i64>,
    pub global_attention_blocks: Vec<usize>,
    pub position_embedding_size: [i64; 2],
    #[setters(generate)]
    pub parameter_dtype: DType,
}

impl HieraConfig {
    /// Official SAM 2.1 Hiera-small trunk configuration.
    pub fn sam2_1_small() -> Self {
        Self {
            embed_dim: 96,
            num_heads: 1,
            stages: vec![1, 2, 11, 2],
            q_pool: 3,
            q_stride: [2, 2],
            window_spec: vec![8, 4, 14, 7],
            global_attention_blocks: vec![7, 10, 13],
            position_embedding_size: [7, 7],
            parameter_dtype: DType::F32,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let valid = self.embed_dim > 0
            && self.num_heads > 0
            && !self.stages.is_empty()
            && self.stages.iter().all(|&depth| depth > 0)
            && self.stages.len() == self.window_spec.len()
            && self.q_pool < self.stages.len()
            && self.q_stride.iter().all(|&stride| stride > 0)
            && self.window_spec.iter().all(|&window| window > 0)
            && self
                .position_embedding_size
                .iter()
                .all(|&dimension| dimension > 0);
        let valid = valid && matches!(self.parameter_dtype, DType::F16 | DType::BF16 | DType::F32);
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidSam2HieraConfiguration,
            })
        }
    }
}

impl Default for HieraConfig {
    fn default() -> Self {
        Self::sam2_1_small()
    }
}
