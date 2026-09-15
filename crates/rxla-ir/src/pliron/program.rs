//! Ownership, identity, ABI, and lowering orchestration for one Pliron program.

use super::*;
use cranelift_entity::{PrimaryMap, entity_impl};

/// Pliron state owned by a graph that is being built directly by the frontend.
/// SSA handles never escape this owner or outlive its Context.
#[derive(Default)]
pub struct ProgramIr {
    pub(super) graph: IrGraph,
    pub(super) values: PrimaryMap<SsaId, Value>,
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
        let Some(parameter) = op.as_ref().downcast_ref::<ParameterOp>() else {
            return Ok(None);
        };
        parameter
            .get_attr_number(&self.graph.ctx)
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
