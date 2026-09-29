use crate::{Error, Result};
use rxla_core::Tensor;

/// One statically shaped slice in a [`MicroBatchPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MicroBatchChunk {
    start: usize,
    length: usize,
    padding: usize,
}

impl MicroBatchChunk {
    pub fn start(self) -> usize {
        self.start
    }

    pub fn length(self) -> usize {
        self.length
    }

    pub fn padding(self) -> usize {
        self.padding
    }
}

/// A reusable static split/pad plan for executing a large logical batch with
/// one smaller compiled model shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicroBatchPlan {
    batch_size: usize,
    micro_batch_size: usize,
    chunks: Vec<MicroBatchChunk>,
}

impl MicroBatchPlan {
    pub fn new(batch_size: usize, max_batch_size: usize) -> Result<Self> {
        if batch_size == 0 || max_batch_size == 0 {
            return Err(Error::InvalidMicroBatchSize {
                batch_size,
                max_batch_size,
            });
        }
        let micro_batch_size = batch_size.min(max_batch_size);
        let chunks = (0..batch_size)
            .step_by(micro_batch_size)
            .map(|start| {
                let length = micro_batch_size.min(batch_size - start);
                MicroBatchChunk {
                    start,
                    length,
                    padding: micro_batch_size - length,
                }
            })
            .collect();
        Ok(Self {
            batch_size,
            micro_batch_size,
            chunks,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn micro_batch_size(&self) -> usize {
        self.micro_batch_size
    }

    pub fn chunks(&self) -> &[MicroBatchChunk] {
        &self.chunks
    }

    /// Split one symbolic tensor and zero-pad the final slice to the compiled
    /// micro-batch shape. Padding stays in IR and therefore on the device.
    pub fn split_zero_padded(&self, input: &Tensor, axis: usize) -> Result<Vec<Tensor>> {
        let Some(&actual) = input.shape().get(axis) else {
            return Err(Error::MicroBatchAxis {
                axis,
                rank: input.shape().len(),
            });
        };
        if usize::try_from(actual).ok() != Some(self.batch_size) {
            return Err(Error::MicroBatchDimension {
                axis,
                actual,
                expected: self.batch_size,
            });
        }
        self.chunks
            .iter()
            .map(|chunk| {
                let value = input.narrow(
                    axis,
                    i64::try_from(chunk.start).map_err(|_| Error::MicroBatchDimension {
                        axis,
                        actual,
                        expected: self.batch_size,
                    })?,
                    i64::try_from(chunk.length).map_err(|_| Error::MicroBatchDimension {
                        axis,
                        actual,
                        expected: self.batch_size,
                    })?,
                )?;
                if chunk.padding == 0 {
                    return Ok(value);
                }
                let mut padding_shape = input.shape().to_vec();
                padding_shape[axis] =
                    i64::try_from(chunk.padding).map_err(|_| Error::MicroBatchDimension {
                        axis,
                        actual,
                        expected: self.batch_size,
                    })?;
                let padding = input
                    .scalar(0.0)?
                    .cast(input.dtype())?
                    .broadcast_to(padding_shape)?;
                Ok(Tensor::concatenate(&[value, padding], axis)?)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cx, Model, ModelInput};

    #[test]
    fn split_zero_pads_only_the_final_micro_batch() {
        let plan = MicroBatchPlan::new(10, 4).unwrap();
        assert_eq!(plan.micro_batch_size(), 4);
        assert_eq!(
            plan.chunks().iter().map(|c| c.length()).collect::<Vec<_>>(),
            [4, 4, 2]
        );
        assert_eq!(plan.chunks()[2].padding(), 2);
        let model = Model::new(|_: Cx, input: Tensor| plan.split_zero_padded(&input, 0))
            .inputs(ModelInput::new([10, 3]))
            .trace()
            .unwrap();
        assert_eq!(
            model
                .outputs()
                .iter()
                .map(Tensor::shape)
                .collect::<Vec<_>>(),
            [&[4, 3], &[4, 3], &[4, 3]]
        );
        model.prepare().unwrap();
    }
}
