//! Explicit cuts in immutable semantic SSA, with a pruned ABI per stage.

use crate::{Compiler, Executable, LoweredProgram, Result, Tensor, Tracer};
use rxla_ir::SemanticProgram;
use std::sync::Arc;

/// Origin of one native execution-stage argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageInput {
    /// An argument in the original program's ABI.
    Original(usize),
    /// A preparation output in the caller's cut order.
    Prepared(usize),
}

/// Two immutable programs produced by cutting selected SSA values.
pub struct StagedProgram {
    preparation: SemanticProgram,
    execution: SemanticProgram,
    original_input_count: usize,
}

pub struct CompiledStages {
    pub preparation: Arc<Executable>,
    pub preparation_inputs: Vec<usize>,
    pub execution: Arc<Executable>,
    pub execution_inputs: Vec<StageInput>,
}

impl Tracer {
    /// Extract preparation outputs and replace their execution uses with typed
    /// inputs. This does not alter the tracer or its original outputs.
    pub fn stage_many(&self, outputs: &[Tensor], cuts: &[Tensor]) -> Result<StagedProgram> {
        if outputs
            .iter()
            .chain(cuts)
            .any(|value| !Arc::ptr_eq(&self.graph.0, &value.graph().0))
        {
            return Err(rxla_ir::IrError::InvalidStage {
                reason: "outputs and cuts must belong to this tracer",
            }
            .into());
        }
        let ir = self
            .graph
            .0
            .lock()
            .map_err(|_| crate::Error::GraphLockPoisoned)?;
        let source = SemanticProgram::capture(
            &ir,
            &outputs.iter().map(Tensor::node_id).collect::<Vec<_>>(),
            false,
        )?;
        let (preparation, execution, original_input_count) =
            source.split_at(&cuts.iter().map(Tensor::node_id).collect::<Vec<_>>())?;
        Ok(StagedProgram {
            preparation,
            execution,
            original_input_count,
        })
    }
}

impl StagedProgram {
    pub fn compile(&self, compiler: &mut Compiler) -> Result<CompiledStages> {
        let target = compiler.lowering_target()?;
        let preparation = self.preparation.lower(target)?;
        let execution = self.execution.lower(target)?;
        let preparation_inputs = preparation.parameters.clone();
        let execution_inputs = execution
            .parameters
            .iter()
            .map(|&number| {
                if number < self.original_input_count {
                    StageInput::Original(number)
                } else {
                    StageInput::Prepared(number - self.original_input_count)
                }
            })
            .collect();
        Ok(CompiledStages {
            preparation: compiler.compile_lowered(&LoweredProgram::from_stablehlo(
                preparation.code,
                preparation.inputs,
                preparation.outputs,
            ))?,
            preparation_inputs,
            execution: compiler.compile_lowered(&LoweredProgram::from_stablehlo(
                execution.code,
                execution.inputs,
                execution.outputs,
            ))?,
            execution_inputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_validates_tensor_provenance_without_a_runtime() -> Result<()> {
        let tracer = Tracer::new();
        let input = tracer.input([4])?;
        let cut = input.exp()?;
        let output = cut.add(&input)?;
        let stages =
            tracer.stage_many(std::slice::from_ref(&output), std::slice::from_ref(&cut))?;
        let target = rxla_ir::LoweringTarget::Portable;
        assert_eq!(stages.preparation.lower(target)?.parameters, [0]);
        assert_eq!(stages.execution.lower(target)?.parameters, [0, 1]);
        let foreign = Tracer::new().input([4])?.exp()?;
        assert!(
            tracer
                .stage_many(
                    std::slice::from_ref(&output),
                    std::slice::from_ref(&foreign)
                )
                .is_err()
        );
        assert!(tracer.stage_many(&[foreign], &[cut]).is_err());
        // The ordinary program remains usable after staging and rejected attempts.
        tracer.program(vec![output])?;
        Ok(())
    }
}
