//! Parameter-aware bindings for two programs extracted from one model SSA.

use crate::*;
use rxla_core::{Buffer, CompiledStages, Compiler, StageInput};
use snafu::{OptionExt, ensure};
use std::sync::Arc;

pub struct CompiledModelStages {
    model: AppliedModel,
    stages: CompiledStages,
    preparation_inputs: Vec<usize>,
    execution_inputs: Vec<usize>,
}

/// Shared weights with separate preparation and execution entry points.
pub struct BoundModelStages {
    compiled: Arc<CompiledModelStages>,
    parameters: Vec<Option<Buffer>>,
    identity: Arc<()>,
}

/// Immutable intermediate buffers prepared for one particular weight binding.
/// Reuse them across invocations while that binding and its inputs are valid.
pub struct PreparedModelValues {
    buffers: Vec<Buffer>,
    identity: Arc<()>,
}

impl AppliedModel {
    /// Cut named intermediates out of the hot execution program. Both stages
    /// derive from the same immutable IR; each gets a pruned parameter/input ABI.
    pub fn compile_staged(
        &self,
        compiler: &mut Compiler,
        paths: &[&str],
    ) -> Result<CompiledModelStages> {
        ensure!(
            !self.is_stateful(),
            StatefulOperationSnafu {
                operation: "compile_staged"
            }
        );
        let cuts = paths
            .iter()
            .map(|path| self.tap(path).cloned())
            .collect::<Result<Vec<_>>>()?;
        let (tracer, outputs) = self.clone().into_parts();
        let stages = tracer.stage_many(&outputs, &cuts)?.compile(compiler)?;
        let model_inputs = |sources: &[usize]| -> Result<Vec<usize>> {
            sources
                .iter()
                .filter_map(|&source| match self.schema().arguments().get(source) {
                    Some(ModelArgument::Input(index)) => Some(Ok(*index)),
                    Some(ModelArgument::Parameter(_)) => None,
                    _ => Some(InvalidStageAbiSnafu.fail()),
                })
                .collect()
        };
        let preparation_inputs = model_inputs(&stages.preparation_inputs)?;
        let execution_inputs = model_inputs(
            &stages
                .execution_inputs
                .iter()
                .filter_map(|input| match input {
                    StageInput::Original(index) => Some(*index),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(CompiledModelStages {
            model: self.clone(),
            stages,
            preparation_inputs,
            execution_inputs,
        })
    }
}

impl CompiledModelStages {
    /// Original model input indices consumed by preparation, in call order.
    pub fn preparation_inputs(&self) -> &[usize] {
        &self.preparation_inputs
    }

    /// Original model input indices consumed by execution, in call order.
    pub fn execution_inputs(&self) -> &[usize] {
        &self.execution_inputs
    }

    pub fn bind_parameters<'a>(
        self,
        parameters: impl IntoIterator<Item = (&'a str, &'a Buffer)>,
    ) -> Result<BoundModelStages> {
        let parameters = self
            .model
            .order_parameters(parameters)?
            .into_iter()
            .map(|buffer| buffer.cloned())
            .collect();
        Ok(BoundModelStages {
            compiled: Arc::new(self),
            parameters,
            identity: Arc::new(()),
        })
    }
}

impl BoundModelStages {
    pub fn preparation_inputs(&self) -> &[usize] {
        self.compiled.preparation_inputs()
    }
    pub fn execution_inputs(&self) -> &[usize] {
        self.compiled.execution_inputs()
    }

    /// Execute preparation using only the inputs its dependency closure needs.
    pub fn prepare<'a, I: ModelInputValues<'a>>(&self, inputs: I) -> Result<PreparedModelValues> {
        let inputs = inputs.into_values();
        self.check_count(inputs.len(), self.preparation_inputs().len())?;
        let mut position = 0;
        let arguments = self
            .compiled
            .stages
            .preparation_inputs
            .iter()
            .map(|&source| self.original_argument(source, &inputs, &mut position))
            .collect::<Result<Vec<_>>>()?;
        let buffers = self.compiled.stages.preparation.execute(&arguments)?;
        Ok(PreparedModelValues {
            buffers,
            identity: self.identity.clone(),
        })
    }

    /// Run the remaining computation with prepared intermediates and only its
    /// remaining dynamic inputs. Values prepared for another binding are rejected.
    pub fn run<'a, I: ModelInputValues<'a>, O: ModelOutputValues>(
        &self,
        inputs: I,
        prepared: &PreparedModelValues,
    ) -> Result<O> {
        ensure!(
            Arc::ptr_eq(&self.identity, &prepared.identity),
            PreparedModelMismatchSnafu
        );
        let inputs = inputs.into_values();
        self.check_count(inputs.len(), self.execution_inputs().len())?;
        let mut position = 0;
        let arguments = self
            .compiled
            .stages
            .execution_inputs
            .iter()
            .map(|source| match *source {
                StageInput::Original(index) => {
                    self.original_argument(index, &inputs, &mut position)
                }
                StageInput::Prepared(index) => {
                    prepared.buffers.get(index).context(InvalidStageAbiSnafu)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.compiled
            .model
            .decode_outputs(self.compiled.stages.execution.execute(&arguments)?)
    }

    fn original_argument<'a>(
        &'a self,
        source: usize,
        inputs: &[&'a Buffer],
        position: &mut usize,
    ) -> Result<&'a Buffer> {
        match self.compiled.model.schema().arguments().get(source) {
            Some(ModelArgument::Input(_)) => {
                let value = inputs
                    .get(*position)
                    .copied()
                    .context(InvalidStageAbiSnafu)?;
                *position += 1;
                Ok(value)
            }
            Some(ModelArgument::Parameter(index)) => self
                .parameters
                .get(*index)
                .and_then(Option::as_ref)
                .context(InvalidStageAbiSnafu),
            _ => InvalidStageAbiSnafu.fail(),
        }
    }

    fn check_count(&self, actual: usize, expected: usize) -> Result<()> {
        ensure!(actual == expected, InputCountSnafu { expected, actual });
        Ok(())
    }
}

impl PreparedModelValues {
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }
}
