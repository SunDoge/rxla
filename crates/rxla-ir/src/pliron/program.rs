//! Ownership, identity, ABI, and lowering orchestration for one Pliron program.

use super::*;
use cranelift_entity::{PrimaryMap, entity_impl};

fn join_ids(values: &[SsaId]) -> String {
    values
        .iter()
        .map(|value| value.index().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Pliron state owned by a graph that is being built directly by the frontend.
/// SSA handles never escape this owner or outlive its Context.
#[derive(Default)]
pub struct ProgramIr {
    pub(super) graph: IrGraph,
    pub(super) values: PrimaryMap<SsaId, Value>,
}

#[cfg(test)]
mod thread_safety_tests {
    use super::ProgramIr;

    fn assert_send<T: Send>() {}

    #[test]
    fn program_ir_is_send() {
        assert_send::<ProgramIr>();
    }
}

/// Graph-local identity of a canonical Pliron SSA value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaId(u32);

entity_impl!(SsaId, "ssa");

impl SsaId {
    pub fn from_index(index: usize) -> Self {
        <Self as cranelift_entity::EntityRef>::new(index)
    }

    pub fn index(self) -> usize {
        <Self as cranelift_entity::EntityRef>::index(self)
    }
}

pub struct StableHloProgram {
    pub code: String,
    pub inputs: Vec<TensorType>,
    pub parameters: Vec<usize>,
    pub outputs: Vec<TensorType>,
}

impl ProgramIr {
    /// Marker used by structured-region builders to identify newly appended
    /// root operations before they are moved into a region.
    pub fn region_marker(&self) -> usize {
        self.values.len()
    }

    /// Build a structured conditional from two already-built
    /// root ranges. The ranges are moved, not cloned, into the corresponding
    /// branch regions; values before each marker remain legal captures.
    pub fn append_conditional(
        &mut self,
        predicate: SsaId,
        then_marker: usize,
        then_values: &[SsaId],
        else_marker: usize,
        else_values: &[SsaId],
        results: &[TensorType],
    ) -> Result<Vec<SsaId>> {
        let predicate = self.value(predicate)?;
        if results.is_empty()
            || then_values.len() != results.len()
            || else_values.len() != results.len()
        {
            return Err(IrError::InvalidValue {
                operation: "building conditional results",
            });
        }
        let then_values_raw = then_values
            .iter()
            .map(|&value| self.value(value))
            .collect::<Result<Vec<_>>>()?;
        let else_values_raw = else_values
            .iter()
            .map(|&value| self.value(value))
            .collect::<Result<Vec<_>>>()?;
        let then_end = else_marker;
        let else_end = self.values.len();
        if then_marker > then_end || then_end > else_end {
            return Err(IrError::InvalidValue {
                operation: "building conditional regions",
            });
        }

        let result_types = results
            .iter()
            .map(|result| self.graph.tensor_type(&result.dims, result.dtype))
            .collect();
        let conditional = <IfOp as PlironOp>::from_operation(Operation::new(
            &mut self.graph.ctx,
            IfOp::get_concrete_op_info(),
            result_types,
            vec![predicate],
            vec![],
            2,
        ));
        conditional.set_attr_then_marker(&self.graph.ctx, StringAttr::new(then_marker.to_string()));
        conditional.set_attr_then_value(&self.graph.ctx, StringAttr::new(join_ids(then_values)));
        conditional.set_attr_else_marker(&self.graph.ctx, StringAttr::new(else_marker.to_string()));
        conditional.set_attr_else_value(&self.graph.ctx, StringAttr::new(join_ids(else_values)));
        for (region_index, (start, end, yielded)) in [
            (then_marker, then_end, then_values_raw),
            (else_marker, else_end, else_values_raw),
        ]
        .into_iter()
        .enumerate()
        {
            let root = self.graph.module.get_body(&self.graph.ctx, 0);
            let block = BasicBlock::new(&mut self.graph.ctx, None, vec![]);
            block.insert_at_back(
                conditional
                    .get_operation()
                    .deref(&self.graph.ctx)
                    .get_region(region_index),
                &self.graph.ctx,
            );
            let mut operations = Vec::new();
            for index in start..end {
                let value = self.value(SsaId::from_index(index))?;
                let operation = value.defining_op().ok_or(IrError::InvalidValue {
                    operation: "moving a conditional branch into its region",
                })?;
                // A nested structured operation has already moved its own
                // branch operations. Move only the nested operation itself;
                // pulling its internals back out would violate region
                // dominance and leave its yields referring outside the region.
                if operation.deref(&self.graph.ctx).get_parent_block() != Some(root) {
                    continue;
                }
                if operations.last().copied() != Some(operation) {
                    operations.push(operation);
                }
            }
            for operation in operations {
                let op = Operation::get_op_dyn(operation, &self.graph.ctx);
                if op.as_ref().is::<ParameterOp>() || op.as_ref().is::<StateInputOp>() {
                    // Effect declarations belong to the function ABI. Region
                    // operations capture their values instead of nesting the
                    // declarations inside one branch.
                    continue;
                }
                operation.unlink(&self.graph.ctx);
                operation.insert_at_back(block, &self.graph.ctx);
            }
            let yield_op = <YieldOp as PlironOp>::from_operation(Operation::new(
                &mut self.graph.ctx,
                YieldOp::get_concrete_op_info(),
                vec![],
                yielded,
                vec![],
                0,
            ));
            yield_op
                .get_operation()
                .insert_at_back(block, &self.graph.ctx);
        }
        Ok(self
            .graph
            .push_results(conditional)
            .into_iter()
            .map(|value| self.values.push(value))
            .collect())
    }

    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Return the canonical parameter ABI width after checking that parameter
    /// numbers form one dense, unique range starting at zero.
    pub fn parameter_count(&self) -> Result<usize> {
        let mut numbers = self
            .values
            .keys()
            .filter_map(|id| self.parameter_number(id).transpose())
            .collect::<Result<Vec<_>>>()?;
        numbers.sort_unstable();
        if numbers.iter().copied().eq(0..numbers.len()) {
            Ok(numbers.len())
        } else {
            Err(IrError::MalformedAttribute {
                attribute: "dense parameter ABI numbering",
            })
        }
    }

    pub fn value_type(&self, id: SsaId) -> Result<TensorType> {
        let value = self.value(id)?;
        let ty = value.get_type(&self.graph.ctx);
        let ty = ty.deref(&self.graph.ctx);
        let ranked = ty
            .downcast_ref::<RankedTensorType>()
            .ok_or(IrError::ExpectedTensorType {
                operation: "reading an SSA value type",
            })?;
        ranked
            .verify(&self.graph.ctx)
            .map_err(|error| IrError::Verification {
                stage: "tensor type query",
                message: error.to_string(),
            })?;
        Ok(TensorType {
            dims: ranked.shape.values(),
            dtype: ranked.element.value(),
        })
    }

    pub fn operand_types(&self, ids: &[SsaId]) -> Result<Vec<DType>> {
        ids.iter()
            .map(|&id| self.value_type(id).map(|ty| ty.dtype))
            .collect()
    }

    pub fn operand_ids(&self, id: SsaId) -> Result<Vec<SsaId>> {
        let operation = self.defining_op(id, "reading SSA operands")?;
        operation
            .deref(&self.graph.ctx)
            .operands()
            .map(|operand| {
                self.values
                    .iter()
                    .find_map(|(id, &value)| (value == operand).then_some(id))
                    .ok_or(IrError::InvalidValue {
                        operation: "indexing an operand",
                    })
            })
            .collect::<std::result::Result<_, IrError>>()
    }

    pub fn parameter_number(&self, id: SsaId) -> Result<Option<usize>> {
        let operation = self.defining_op(id, "reading a parameter ABI number")?;
        let op = Operation::get_op_dyn(operation, &self.graph.ctx);
        let number = if let Some(parameter) = op.as_ref().downcast_ref::<ParameterOp>() {
            parameter.get_attr_number(&self.graph.ctx)
        } else if let Some(state) = op.as_ref().downcast_ref::<StateInputOp>() {
            state.get_attr_state_number(&self.graph.ctx)
        } else {
            return Ok(None);
        };
        number
            .ok_or(IrError::MalformedAttribute {
                attribute: "parameter ABI number",
            })?
            .as_str()
            .parse()
            .map(Some)
            .map_err(|_| IrError::MalformedAttribute {
                attribute: "parameter ABI number",
            })
    }

    /// Append one logical state input effect.
    pub fn state_input(
        &mut self,
        number: usize,
        state_id: usize,
        path: &str,
        ty: &TensorType,
    ) -> Result<SsaId> {
        let value = self.graph.state_input(number, state_id, path, ty);
        Ok(self.values.push(value))
    }

    /// Append a read effect for the current state SSA version.
    pub fn state_read(&mut self, current: SsaId, state_id: usize) -> Result<SsaId> {
        let current = self.value(current)?;
        let value = self.graph.state_read(current, state_id);
        Ok(self.values.push(value))
    }

    /// Append a write effect and return the next state SSA version.
    pub fn state_write(&mut self, value: SsaId, state_id: usize) -> Result<SsaId> {
        let value = self.value(value)?;
        let next = self.graph.state_write(value, state_id);
        Ok(self.values.push(next))
    }

    pub fn replace_with_parameter(
        &mut self,
        id: SsaId,
        number: usize,
        ty: &TensorType,
    ) -> Result<()> {
        self.index(id)?;
        let old = self.values[id];
        let new = self.graph.parameter_number(number, &ty.dims, ty.dtype);
        old.replace_all_uses_with(&self.graph.ctx, &new);
        self.values[id] = new;
        Ok(())
    }

    fn index(&self, id: SsaId) -> Result<usize> {
        self.values
            .is_valid(id)
            .then_some(id.index())
            .ok_or(IrError::InvalidValue {
                operation: "resolving a tensor ID",
            })
    }

    fn value(&self, id: SsaId) -> Result<Value> {
        self.index(id)?;
        Ok(self.values[id])
    }

    fn defining_op(
        &self,
        id: SsaId,
        operation: &'static str,
    ) -> Result<pliron::context::Ptr<Operation>> {
        self.value(id)?
            .defining_op()
            .ok_or(IrError::InvalidValue { operation })
    }

    pub fn stablehlo(&mut self, outputs: &[SsaId]) -> Result<String> {
        let outputs = self.resolve_outputs(outputs).ok_or(IrError::InvalidValue {
            operation: "resolving StableHLO outputs",
        })?;
        self.graph.export_stablehlo(&outputs, false)
    }

    pub fn stablehlo_program(
        &mut self,
        outputs: &[SsaId],
        preserve_all_inputs: bool,
    ) -> Result<StableHloProgram> {
        self.stablehlo_program_for(outputs, preserve_all_inputs, LoweringTarget::Portable)
    }

    pub fn stablehlo_program_for(
        &mut self,
        outputs: &[SsaId],
        preserve_all_inputs: bool,
        target: LoweringTarget,
    ) -> Result<StableHloProgram> {
        let outputs = self.resolve_outputs(outputs).ok_or(IrError::InvalidValue {
            operation: "resolving StableHLO outputs",
        })?;
        self.graph.verify("source Pliron")?;
        self.graph.ensure_stablehlo_supported(&outputs)?;
        let (inputs, parameters, output_types) = self
            .graph
            .stablehlo_signature(&outputs, preserve_all_inputs)?;
        let code = self
            .graph
            .export_stablehlo_for(&outputs, preserve_all_inputs, target)?;
        Ok(StableHloProgram {
            code,
            inputs,
            parameters,
            outputs: output_types,
        })
    }

    pub fn planning_snapshot(&self, outputs: &[SsaId]) -> Result<PlanningFacts> {
        self.graph.verify("source Pliron planning")?;
        let outputs = self.resolve_outputs(outputs).ok_or(IrError::InvalidValue {
            operation: "resolving planning outputs",
        })?;
        self.graph.planning_snapshot(&outputs)
    }

    fn resolve_outputs(&self, outputs: &[SsaId]) -> Option<Vec<Value>> {
        outputs
            .iter()
            .map(|&id| self.values.get(id).copied())
            .collect()
    }

    pub fn set_sharding(&mut self, id: SsaId, sharding: &Sharding) -> Result<()> {
        let value = self.values.get(id).copied().ok_or(IrError::InvalidValue {
            operation: "attaching a sharding constraint",
        })?;
        let operation = value.defining_op().ok_or(IrError::InvalidValue {
            operation: "attaching a sharding constraint",
        })?;
        if let Some(existing) = operation
            .deref(&self.graph.ctx)
            .attributes
            .get::<ShardingAttr>(&sharding_attr_key())
            && &existing.sharding() != sharding
        {
            return Err(IrError::ShardingConflict);
        }
        operation
            .deref_mut(&self.graph.ctx)
            .attributes
            .set(sharding_attr_key(), ShardingAttr::new(sharding)?);
        Ok(())
    }

    pub fn sharding(&self, id: SsaId) -> Result<Option<Sharding>> {
        let operation = self.defining_op(id, "reading a sharding constraint")?;
        Ok(operation
            .deref(&self.graph.ctx)
            .attributes
            .get::<ShardingAttr>(&sharding_attr_key())
            .map(|attribute| attribute.sharding()))
    }
}
