//! Serialization of slicing, padding, concatenation, and dynamic indexing.

use super::emission_support::{OperationEmission, stablehlo_type, value_type};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_basic_indexing(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let operation_ptr = emission.operation;
    let operation = operation_ptr.deref(ctx);
    let op = Operation::get_op_dyn(operation_ptr, ctx);
    if let Some(slice) = op.as_ref().downcast_ref::<StableSliceOp>() {
        let source = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let ranges = slice
            .get_attr_stable_slice_spec(ctx)
            .unwrap()
            .values()
            .iter()
            .map(|axis| format!("{}:{}:{}", axis.start, axis.limit, axis.stride))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.slice {} [{ranges}] : ({source}) -> {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else if let Some(gradient) = op.as_ref().downcast_ref::<StableSliceGradientOp>() {
        let input = value_type(ctx, operation.get_operand(0));
        let input_type = stablehlo_type(&input)?;
        let zero = emission.auxiliary();
        writeln!(
            emission.body,
            "    {zero} = stablehlo.constant dense<0x00000000> : tensor<f32>"
        )
        .unwrap();
        if input.dims.contains(&0) {
            writeln!(
                emission.body,
                "    {} = stablehlo.broadcast_in_dim {zero}, dims = [] : (tensor<f32>) -> {}",
                emission.name, emission.ty
            )
            .unwrap();
        } else {
            let spec = gradient
                .get_attr_stable_gradient_spec(ctx)
                .unwrap()
                .values();
            let low = spec
                .iter()
                .map(|axis| axis.start.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let high = spec
                .iter()
                .enumerate()
                .map(|(dimension, axis)| {
                    (emission.result_type.dims[dimension]
                        - axis.start
                        - 1
                        - (input.dims[dimension] - 1) * axis.stride)
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(", ");
            let interior = spec
                .iter()
                .map(|axis| (axis.stride - 1).to_string())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(emission.body, "    {} = stablehlo.pad {}, {zero}, low = [{low}], high = [{high}], interior = [{interior}] : ({input_type}, tensor<f32>) -> {}", emission.name, emission.operands[0], emission.ty).unwrap();
        }
    } else if let Some(pad) = op.as_ref().downcast_ref::<StablePadOp>() {
        let input_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let fill_type = stablehlo_type(&value_type(ctx, operation.get_operand(1)))?;
        let padding = pad.get_attr_stable_padding(ctx).unwrap().values();
        let list = |index: usize| {
            padding
                .iter()
                .map(|axis| axis[index].to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let interior = std::iter::repeat_n("0", padding.len())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(emission.body, "    {} = stablehlo.pad {}, {}, low = [{}], high = [{}], interior = [{interior}] : ({input_type}, {fill_type}) -> {}", emission.name, emission.operands[0], emission.operands[1], list(0), list(1), emission.ty).unwrap();
    } else if let Some(concatenate) = op.as_ref().downcast_ref::<StableConcatenateOp>() {
        let axis = concatenate
            .get_attr_stable_concatenate_axis(ctx)
            .unwrap()
            .value();
        let input_types = operation
            .operands()
            .map(|operand| stablehlo_type(&value_type(ctx, operand)))
            .collect::<Result<Vec<_>>>()?
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.concatenate {}, dim = {axis} : ({input_types}) -> {}",
            emission.name,
            emission.operands.join(", "),
            emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableDynamicSliceOp>() {
        let operand_types = operation
            .operands()
            .map(|operand| stablehlo_type(&value_type(ctx, operand)))
            .collect::<Result<Vec<_>>>()?
            .join(", ");
        let sizes = emission
            .result_type
            .dims
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.dynamic_slice {}, sizes = [{sizes}] : ({operand_types}) -> {}",
            emission.name,
            emission.operands.join(", "),
            emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableDynamicUpdateSliceOp>() {
        let operand_types = operation
            .operands()
            .map(|operand| stablehlo_type(&value_type(ctx, operand)))
            .collect::<Result<Vec<_>>>()?
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.dynamic_update_slice {} : ({operand_types}) -> {}",
            emission.name,
            emission.operands.join(", "),
            emission.ty
        )
        .unwrap();
    } else {
        return Ok(false);
    }
    Ok(true)
}
