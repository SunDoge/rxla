//! Typed StableHLO lowering followed by textual MLIR serialization for PJRT.

mod aggregation_emitter;
mod attention_emitter;
mod conversion;
mod convolution_emitter;
mod dialect;
mod emission_support;
mod emitter;
mod gather_emitter;
mod indexing_emitter;
mod ordered_reduction_emitter;
mod predicate_shape_emitter;
mod primitive_emitter;

use super::*;
use conversion::{lower_to_stablehlo, supports_source_operation};
use dialect::ExportOp;
use pliron::context::Ptr;
use pliron::irbuild::{
    cloning::{IrMapping, clone_operation},
    listener::DummyListener,
    rewriter::{IRRewriter, Rewriter},
};

impl IrGraph {
    pub(super) fn ensure_stablehlo_supported(&self, outputs: &[Value]) -> Result<()> {
        let reachable = self.reachable_values(outputs)?;
        for operation in self.operations() {
            let result = operation.deref(&self.ctx).get_result(0);
            if reachable.contains(&result)
                && !supports_source_operation(Operation::get_op_dyn(operation, &self.ctx).as_ref())
            {
                return Err(IrError::UnsupportedOperation {
                    operation: Operation::get_opid(operation, &self.ctx).to_string(),
                });
            }
        }
        Ok(())
    }

    pub(super) fn stablehlo_signature(
        &self,
        outputs: &[Value],
        preserve_all_inputs: bool,
    ) -> Result<(Vec<TensorType>, Vec<usize>, Vec<TensorType>)> {
        let reachable = self.reachable_values(outputs)?;
        let mut inputs = Vec::new();
        for operation in self.operations() {
            let result = operation.deref(&self.ctx).get_result(0);
            if !preserve_all_inputs && !reachable.contains(&result) {
                continue;
            }
            let op = Operation::get_op_dyn(operation, &self.ctx);
            let number = if let Some(parameter) = op.as_ref().downcast_ref::<ParameterOp>() {
                parameter.get_attr_number(&self.ctx)
            } else if let Some(state) = op.as_ref().downcast_ref::<StateInputOp>() {
                state.get_attr_state_number(&self.ctx)
            } else {
                continue;
            };
            let number = number
                .ok_or(IrError::MalformedAttribute {
                    attribute: "parameter ABI number",
                })?
                .as_str()
                .parse()
                .map_err(|_| IrError::MalformedAttribute {
                    attribute: "parameter ABI number",
                })?;
            inputs.push((number, self.value_type(result)?));
        }
        inputs.sort_unstable_by_key(|(number, _)| *number);
        if inputs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(IrError::MalformedAttribute {
                attribute: "unique parameter ABI numbers",
            });
        }
        let parameters = inputs.iter().map(|(number, _)| *number).collect();
        let inputs = inputs.into_iter().map(|(_, ty)| ty).collect();
        let outputs = outputs
            .iter()
            .map(|output| self.value_type(*output))
            .collect::<Result<Vec<_>>>()?;
        Ok((inputs, parameters, outputs))
    }

    pub(super) fn export_stablehlo(
        &mut self,
        outputs: &[Value],
        preserve_all_inputs: bool,
    ) -> Result<String> {
        self.export_stablehlo_for(outputs, preserve_all_inputs, LoweringTarget::Portable)
    }

    pub(super) fn export_stablehlo_for(
        &mut self,
        outputs: &[Value],
        preserve_all_inputs: bool,
        target: LoweringTarget,
    ) -> Result<String> {
        self.verify("source Pliron")?;
        let (boundary_inputs, _, boundary_outputs) =
            self.stablehlo_signature(outputs, preserve_all_inputs)?;
        let mut mapping = IrMapping::new();
        let mut rewriter = IRRewriter::<DummyListener>::default();
        let cloned = clone_operation(
            self.module.get_operation(),
            &mut self.ctx,
            &mut rewriter,
            &mut mapping,
        );
        let cloned_outputs = outputs
            .iter()
            .map(|value| {
                mapping.lookup_value(*value).ok_or(IrError::InvalidValue {
                    operation: "mapping an output into cloned IR",
                })
            })
            .collect::<std::result::Result<Vec<_>, IrError>>()?;
        // Give output values real uses so dialect conversion can update them
        // when it replaces their defining operations.
        let export = <ExportOp as PlironOp>::from_operation(Operation::new(
            &mut self.ctx,
            ExportOp::get_concrete_op_info(),
            vec![],
            cloned_outputs,
            vec![],
            0,
        ));
        Operation::get_op::<ModuleOp>(cloned, &self.ctx)
            .expect("cloned root is a module")
            .append_operation(&mut self.ctx, export.get_operation(), 0);
        let result = (|| {
            lower_to_stablehlo(&mut self.ctx, cloned, target).map_err(|error| {
                IrError::Conversion {
                    message: error.to_string(),
                }
            })?;
            let cloned_module =
                Operation::get_op::<ModuleOp>(cloned, &self.ctx).expect("cloned root is a module");
            verify_op(&cloned_module, &self.ctx).map_err(|error| IrError::Verification {
                stage: "lowered StableHLO",
                message: error.to_string(),
            })?;
            let lowered_outputs = export
                .get_operation()
                .deref(&self.ctx)
                .operands()
                .collect::<Vec<_>>();
            if let Some(dtype) = match target {
                LoweringTarget::CudaF16Compute => Some(DType::F16),
                LoweringTarget::CudaBf16Compute => Some(DType::BF16),
                _ => None,
            } {
                demote_float_results(&self.ctx, cloned, dtype);
            }
            emitter::emit_module(
                &self.ctx,
                cloned,
                &lowered_outputs,
                preserve_all_inputs,
                target,
                &boundary_inputs,
                &boundary_outputs,
            )
        })();
        rewriter.erase_operation(&mut self.ctx, cloned);
        result
    }
}

fn demote_float_results(ctx: &Context, root: Ptr<Operation>, dtype: DType) {
    fn collect(ctx: &Context, operation: Ptr<Operation>, result: &mut Vec<Ptr<Operation>>) {
        result.push(operation);
        let regions = operation.deref(ctx).num_regions();
        for region_index in 0..regions {
            let region = operation.deref(ctx).get_region(region_index);
            for block in region.deref(ctx).iter(ctx) {
                for nested in block.deref(ctx).iter(ctx) {
                    collect(ctx, nested, result);
                }
            }
        }
    }

    let mut operations = Vec::new();
    collect(ctx, root, &mut operations);
    for operation in operations {
        let results = operation.deref(ctx).results().collect::<Vec<_>>();
        for result in results {
            let ty = emission_support::value_type(ctx, result);
            if ty.dtype == DType::F32 {
                result.set_type(
                    ctx,
                    RankedTensorType::get(
                        ctx,
                        ShapeAttr::new(&ty.dims),
                        ElementTypeAttr::new(dtype),
                        DynamicBoundsAttr::new(&ty.dynamic_bounds),
                    )
                    .into(),
                );
            }
        }
    }
}
