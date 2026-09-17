//! Serialization of predicates, selection, and metadata-only shape transforms.

use super::emission_support::{
    OperationEmission, stablehlo_element_type, stablehlo_predicate_type, stablehlo_type, value_type,
};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_predicate_or_shape(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let operation_ptr = emission.operation;
    let operation = operation_ptr.deref(ctx);
    let op = Operation::get_op_dyn(operation_ptr, ctx);
    if let Some(compare) = op.as_ref().downcast_ref::<StableCompareMaskOp>() {
        let lhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let rhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(1)))?;
        let predicate_type = stablehlo_predicate_type(emission.result_type);
        let predicate = emission.auxiliary();
        let comparison_type = if value_type(ctx, operation.get_operand(0)).dtype == DType::I32 {
            "SIGNED"
        } else {
            "FLOAT"
        };
        let direction = compare.get_attr_stable_comparison(ctx).unwrap().direction();
        writeln!(emission.body, "    {predicate} = stablehlo.compare {direction}, {}, {}, {comparison_type} : ({lhs_type}, {rhs_type}) -> {predicate_type}", emission.operands[0], emission.operands[1]).unwrap();
        writeln!(
            emission.body,
            "    {} = stablehlo.convert {predicate} : ({predicate_type}) -> {}",
            emission.name, emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableIsFiniteMaskOp>() {
        let source_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let predicate_type = stablehlo_predicate_type(emission.result_type);
        let predicate = emission.auxiliary();
        writeln!(
            emission.body,
            "    {predicate} = stablehlo.is_finite {} : ({source_type}) -> {predicate_type}",
            emission.operands[0]
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {} = stablehlo.convert {predicate} : ({predicate_type}) -> {}",
            emission.name, emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableSelectOp>() {
        let mask_type = value_type(ctx, operation.get_operand(0));
        let mask_tensor_type = stablehlo_type(&mask_type)?;
        let predicate_type = stablehlo_predicate_type(&mask_type);
        let element_type = stablehlo_element_type(mask_type.dtype)?;
        let zero = emission.auxiliary();
        let zeros = emission.auxiliary();
        let predicate = emission.auxiliary();
        writeln!(
            emission.body,
            "    {zero} = stablehlo.constant dense<0x00000000> : tensor<{element_type}>"
        )
        .unwrap();
        writeln!(emission.body, "    {zeros} = stablehlo.broadcast_in_dim {zero}, dims = [] : (tensor<{element_type}>) -> {mask_tensor_type}").unwrap();
        writeln!(emission.body, "    {predicate} = stablehlo.compare NE, {}, {zeros}, FLOAT : ({mask_tensor_type}, {mask_tensor_type}) -> {predicate_type}", emission.operands[0]).unwrap();
        writeln!(
            emission.body,
            "    {} = stablehlo.select {predicate}, {}, {} : {predicate_type}, {}",
            emission.name, emission.operands[1], emission.operands[2], emission.ty
        )
        .unwrap();
    } else if let Some(broadcast) = op.as_ref().downcast_ref::<StableBroadcastOp>() {
        let source = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let dimensions = broadcast
            .get_attr_stable_broadcast_axes(ctx)
            .unwrap()
            .values()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.broadcast_in_dim {}, dims = [{dimensions}] : ({source}) -> {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else if let Some(transpose) = op.as_ref().downcast_ref::<StableTransposeOp>() {
        let source = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let dimensions = transpose
            .get_attr_stable_permutation(ctx)
            .unwrap()
            .values()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.transpose {}, dims = [{dimensions}] : ({source}) -> {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else if let Some(reverse) = op.as_ref().downcast_ref::<StableReverseOp>() {
        let dimensions = reverse
            .get_attr_stable_reverse_axes(ctx)
            .unwrap()
            .values()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emission.body,
            "    {} = stablehlo.reverse {}, dims = [{dimensions}] : {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else if let Some(dimension) = op.as_ref().downcast_ref::<StableGetDimensionSizeOp>() {
        let source = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let axis = dimension
            .get_attr_stable_dimension_size_axis(ctx)
            .unwrap()
            .value();
        writeln!(
            emission.body,
            "    {} = \"stablehlo.get_dimension_size\"({}) {{dimension = {axis} : i64}} : ({source}) -> {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else {
        return Ok(false);
    }
    Ok(true)
}
