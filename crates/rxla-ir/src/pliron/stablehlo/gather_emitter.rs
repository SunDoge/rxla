//! Serialization of gather-style indexing and its scatter-add adjoint.

use super::emission_support::{OperationEmission, stablehlo_type, value_type};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_gather(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let operation_ptr = emission.operation;
    let operation = operation_ptr.deref(ctx);
    let op = Operation::get_op_dyn(operation_ptr, ctx);
    if let Some(take) = op.as_ref().downcast_ref::<StableTakeOp>() {
        let source = value_type(ctx, operation.get_operand(0));
        let indices = value_type(ctx, operation.get_operand(1));
        let source_type = stablehlo_type(&source)?;
        let indices_type = stablehlo_type(&indices)?;
        let axis = take.get_attr_stable_take_axis(ctx).unwrap().value();
        let offset_dims = (0..emission.result_type.dims.len())
            .filter(|&dimension| dimension < axis || dimension >= axis + indices.dims.len())
            .map(|dimension| dimension.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut slice_sizes = source.dims.clone();
        slice_sizes[axis] = 1;
        let slice_sizes = slice_sizes
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(emission.body, "    {} = \"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<offset_dims = [{offset_dims}], collapsed_slice_dims = [{axis}], start_index_map = [{axis}], index_vector_dim = {}>, indices_are_sorted = false, slice_sizes = array<i64: {slice_sizes}>}}> : ({source_type}, {indices_type}) -> {}", emission.name, emission.operands[0], emission.operands[1], indices.dims.len(), emission.ty).unwrap();
    } else if let Some(take) = op.as_ref().downcast_ref::<StableTakeAlongAxisOp>() {
        let source = value_type(ctx, operation.get_operand(0));
        let indices = value_type(ctx, operation.get_operand(1));
        let source_type = stablehlo_type(&source)?;
        let indices_type = stablehlo_type(&indices)?;
        let axis = take.get_attr_stable_take_along_axis(ctx).unwrap().value();
        let batch_dims = (0..source.dims.len())
            .filter(|&dimension| dimension != axis)
            .map(|dimension| dimension.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let slice_sizes = std::iter::repeat_n("1", source.dims.len())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(emission.body, "    {} = \"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [{axis}], operand_batching_dims = [{batch_dims}], start_indices_batching_dims = [{batch_dims}], start_index_map = [{axis}], index_vector_dim = {}>, indices_are_sorted = false, slice_sizes = array<i64: {slice_sizes}>}}> : ({source_type}, {indices_type}) -> {}", emission.name, emission.operands[0], emission.operands[1], indices.dims.len(), emission.ty).unwrap();
    } else if let Some(gather) = op.as_ref().downcast_ref::<StableGatherGradientOp>() {
        emit_gather_gradient(emission, gather, &operation)?;
    } else {
        return Ok(false);
    }
    Ok(true)
}

fn emit_gather_gradient(
    emission: &mut OperationEmission<'_>,
    gather: &StableGatherGradientOp,
    operation: &Operation,
) -> Result<()> {
    let ctx = emission.ctx;
    let gradient = value_type(ctx, operation.get_operand(0));
    let indices = value_type(ctx, operation.get_operand(1));
    let gradient_type = stablehlo_type(&gradient)?;
    let indices_type = stablehlo_type(&indices)?;
    let (axis, batched) = gather
        .get_attr_stable_gather_gradient(ctx)
        .unwrap()
        .values();
    let scalar_zero = emission.auxiliary();
    let lower = emission.auxiliary();
    let scalar_upper = emission.auxiliary();
    let upper = emission.auxiliary();
    let bounded_low = emission.auxiliary();
    let bounded = emission.auxiliary();
    let scalar_base = emission.auxiliary();
    let base = emission.auxiliary();
    let upper_bound = (emission.result_type.dims[axis] - 1).min(i32::MAX as i64);
    writeln!(
        emission.body,
        "    {scalar_zero} = stablehlo.constant dense<0> : tensor<i32>"
    )
    .unwrap();
    writeln!(emission.body, "    {lower} = stablehlo.broadcast_in_dim {scalar_zero}, dims = [] : (tensor<i32>) -> {indices_type}").unwrap();
    writeln!(
        emission.body,
        "    {scalar_upper} = stablehlo.constant dense<{upper_bound}> : tensor<i32>"
    )
    .unwrap();
    writeln!(emission.body, "    {upper} = stablehlo.broadcast_in_dim {scalar_upper}, dims = [] : (tensor<i32>) -> {indices_type}").unwrap();
    writeln!(
        emission.body,
        "    {bounded_low} = stablehlo.maximum {}, {lower} : {indices_type}",
        emission.operands[1]
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {bounded} = stablehlo.minimum {bounded_low}, {upper} : {indices_type}"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {scalar_base} = stablehlo.constant dense<0x00000000> : tensor<f32>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {base} = stablehlo.broadcast_in_dim {scalar_base}, dims = [] : (tensor<f32>) -> {}",
        emission.ty
    )
    .unwrap();
    let update_window_dims = if batched {
        String::new()
    } else {
        (0..gradient.dims.len())
            .filter(|&dimension| dimension < axis || dimension >= axis + indices.dims.len())
            .map(|dimension| dimension.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let batch_fields = if batched {
        let dimensions = (0..emission.result_type.dims.len())
            .filter(|&dimension| dimension != axis)
            .map(|dimension| dimension.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "input_batching_dims = [{dimensions}], scatter_indices_batching_dims = [{dimensions}], "
        )
    } else {
        String::new()
    };
    writeln!(emission.body, "    {} = \"stablehlo.scatter\"({base}, {bounded}, {}) <{{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [{update_window_dims}], inserted_window_dims = [{axis}], {batch_fields}scatter_dims_to_operand_dims = [{axis}], index_vector_dim = {}>, unique_indices = false}}> ({{", emission.name, emission.operands[0], indices.dims.len()).unwrap();
    writeln!(
        emission.body,
        "    ^bb0(%scatter_lhs: tensor<f32>, %scatter_rhs: tensor<f32>):"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      %scatter_sum = stablehlo.add %scatter_lhs, %scatter_rhs : tensor<f32>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      stablehlo.return %scatter_sum : tensor<f32>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    }}) : ({}, {indices_type}, {gradient_type}) -> {}",
        emission.ty, emission.ty
    )
    .unwrap();
    Ok(())
}
