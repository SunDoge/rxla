//! Serialization of ordered and index-carrying reduction regions.

use super::emission_support::{OperationEmission, stablehlo_type, value_type};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_ordered_reduction(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let op = Operation::get_op_dyn(emission.operation, ctx);
    if let Some(cumsum) = op.as_ref().downcast_ref::<StableCumsumOp>() {
        emit_cumsum(emission, cumsum)?;
    } else if let Some(argmax) = op.as_ref().downcast_ref::<StableArgMaxOp>() {
        emit_argmax(emission, argmax)?;
    } else if let Some(sort) = op.as_ref().downcast_ref::<StableSortedIndicesOp>() {
        emit_sort(emission, sort)?;
    } else {
        return Ok(false);
    }
    Ok(true)
}

fn emit_cumsum(emission: &mut OperationEmission<'_>, cumsum: &StableCumsumOp) -> Result<()> {
    let operation = emission.operation.deref(emission.ctx);
    let source_type = stablehlo_type(&value_type(emission.ctx, operation.get_operand(0)))?;
    let axis = cumsum
        .get_attr_stable_cumsum_axis(emission.ctx)
        .unwrap()
        .value();
    let initializer = emission.auxiliary();
    writeln!(
        emission.body,
        "    {initializer} = stablehlo.constant dense<0x00000000> : tensor<f32>"
    )
    .unwrap();
    let window = emission
        .result_type
        .dims
        .iter()
        .enumerate()
        .map(|(dimension, &length)| if dimension == axis { length } else { 1 })
        .map(|dimension| dimension.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let padding = emission
        .result_type
        .dims
        .iter()
        .enumerate()
        .map(|(dimension, &length)| {
            if dimension == axis {
                format!("[{}, 0]", length - 1)
            } else {
                "[0, 0]".into()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let ones = std::iter::repeat_n("1", emission.result_type.dims.len())
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(emission.body, "    {} = \"stablehlo.reduce_window\"({}, {initializer}) <{{base_dilations = array<i64: {ones}>, padding = dense<[{padding}]> : tensor<{}x2xi64>, window_dilations = array<i64: {ones}>, window_dimensions = array<i64: {window}>, window_strides = array<i64: {ones}>}}> ({{", emission.name, emission.operands[0], emission.result_type.dims.len()).unwrap();
    writeln!(
        emission.body,
        "    ^bb0(%cumsum_lhs: tensor<f32>, %cumsum_rhs: tensor<f32>):"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      %cumsum_sum = stablehlo.add %cumsum_lhs, %cumsum_rhs : tensor<f32>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      stablehlo.return %cumsum_sum : tensor<f32>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    }}) : ({source_type}, tensor<f32>) -> {}",
        emission.ty
    )
    .unwrap();
    Ok(())
}

fn emit_argmax(emission: &mut OperationEmission<'_>, argmax: &StableArgMaxOp) -> Result<()> {
    let operation = emission.operation.deref(emission.ctx);
    let input = value_type(emission.ctx, operation.get_operand(0));
    let input_type = stablehlo_type(&input)?;
    let index_type = stablehlo_type(&TensorType {
        dims: input.dims.clone(),
        dtype: DType::I32,
        dynamic_bounds: input.dynamic_bounds.clone(),
    })?;
    let axis = argmax
        .get_attr_stable_argmax_axis(emission.ctx)
        .unwrap()
        .value();
    let indices = emission.auxiliary();
    let negative_infinity = emission.auxiliary();
    let zero = emission.auxiliary();
    let reduced = emission.auxiliary();
    writeln!(
        emission.body,
        "    {indices} = stablehlo.iota dim = {axis} : {index_type}"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {negative_infinity} = stablehlo.constant dense<0xFF800000> : tensor<f32>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {zero} = stablehlo.constant dense<0> : tensor<i32>"
    )
    .unwrap();
    let reduced_float_type = stablehlo_type(&TensorType {
        dims: emission.result_type.dims.clone(),
        dtype: DType::F32,
        dynamic_bounds: emission.result_type.dynamic_bounds.clone(),
    })?;
    writeln!(emission.body, "    {reduced}:2 = stablehlo.reduce({} init: {negative_infinity}), ({indices} init: {zero}) across dimensions = [{axis}] : ({input_type}, {index_type}, tensor<f32>, tensor<i32>) -> ({reduced_float_type}, {})", emission.operands[0], emission.ty).unwrap();
    writeln!(emission.body, "     reducer(%arg_value_lhs: tensor<f32>, %arg_value_rhs: tensor<f32>) (%arg_index_lhs: tensor<i32>, %arg_index_rhs: tensor<i32>) {{").unwrap();
    writeln!(emission.body, "      %arg_gt = stablehlo.compare GT, %arg_value_lhs, %arg_value_rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>").unwrap();
    writeln!(emission.body, "      %arg_nan = stablehlo.compare NE, %arg_value_lhs, %arg_value_lhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>").unwrap();
    writeln!(
        emission.body,
        "      %arg_value_wins = stablehlo.or %arg_gt, %arg_nan : tensor<i1>"
    )
    .unwrap();
    writeln!(emission.body, "      %arg_eq = stablehlo.compare EQ, %arg_value_lhs, %arg_value_rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>").unwrap();
    writeln!(emission.body, "      %arg_index_lt = stablehlo.compare LT, %arg_index_lhs, %arg_index_rhs, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>").unwrap();
    writeln!(
        emission.body,
        "      %arg_tie_wins = stablehlo.and %arg_eq, %arg_index_lt : tensor<i1>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      %arg_index_wins = stablehlo.or %arg_value_wins, %arg_tie_wins : tensor<i1>"
    )
    .unwrap();
    writeln!(emission.body, "      %arg_value = stablehlo.select %arg_value_wins, %arg_value_lhs, %arg_value_rhs : tensor<i1>, tensor<f32>").unwrap();
    writeln!(emission.body, "      %arg_index = stablehlo.select %arg_index_wins, %arg_index_lhs, %arg_index_rhs : tensor<i1>, tensor<i32>").unwrap();
    writeln!(
        emission.body,
        "      stablehlo.return %arg_value, %arg_index : tensor<f32>, tensor<i32>"
    )
    .unwrap();
    writeln!(emission.body, "    }}").unwrap();
    writeln!(
        emission.body,
        "    {} = stablehlo.reshape {reduced}#1 : ({}) -> {}",
        emission.name, emission.ty, emission.ty
    )
    .unwrap();
    Ok(())
}

fn emit_sort(emission: &mut OperationEmission<'_>, sort: &StableSortedIndicesOp) -> Result<()> {
    let operation = emission.operation.deref(emission.ctx);
    let input = value_type(emission.ctx, operation.get_operand(0));
    let input_type = stablehlo_type(&input)?;
    let (axis, descending) = sort.get_attr_stable_sort(emission.ctx).unwrap().values();
    let indices = emission.auxiliary();
    let sorted = emission.auxiliary();
    let direction = if descending { "GT" } else { "LT" };
    writeln!(
        emission.body,
        "    {indices} = stablehlo.iota dim = {axis} : {}",
        emission.ty
    )
    .unwrap();
    writeln!(emission.body, "    {sorted}:2 = \"stablehlo.sort\"({}, {indices}) <{{dimension = {axis} : i64, is_stable = true}}> ({{", emission.operands[0]).unwrap();
    writeln!(emission.body, "    ^bb0(%sort_lhs: tensor<f32>, %sort_rhs: tensor<f32>, %sort_lhs_index: tensor<i32>, %sort_rhs_index: tensor<i32>):").unwrap();
    writeln!(emission.body, "      %sort_lhs_not_nan = stablehlo.compare EQ, %sort_lhs, %sort_lhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>").unwrap();
    writeln!(emission.body, "      %sort_rhs_nan = stablehlo.compare NE, %sort_rhs, %sort_rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>").unwrap();
    writeln!(emission.body, "      %sort_ordered = stablehlo.compare {direction}, %sort_lhs, %sort_rhs, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>").unwrap();
    writeln!(
        emission.body,
        "      %sort_rhs_nan_or_ordered = stablehlo.or %sort_rhs_nan, %sort_ordered : tensor<i1>"
    )
    .unwrap();
    writeln!(emission.body, "      %sort_before = stablehlo.and %sort_lhs_not_nan, %sort_rhs_nan_or_ordered : tensor<i1>").unwrap();
    writeln!(
        emission.body,
        "      stablehlo.return %sort_before : tensor<i1>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    }}) : ({input_type}, {}) -> ({input_type}, {})",
        emission.ty, emission.ty
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {} = stablehlo.reshape {sorted}#1 : ({}) -> {}",
        emission.name, emission.ty, emission.ty
    )
    .unwrap();
    Ok(())
}
