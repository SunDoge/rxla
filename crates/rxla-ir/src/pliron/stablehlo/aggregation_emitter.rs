//! Serialization of pooling windows and scalar reduction regions.

use super::emission_support::{
    OperationEmission, floating_initializer, stablehlo_element_type, stablehlo_type, value_type,
};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_aggregation(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let operation_ptr = emission.operation;
    let operation = operation_ptr.deref(ctx);
    let op = Operation::get_op_dyn(operation_ptr, ctx);
    if op.as_ref().is::<StableReduceWindowMaximumOp>()
        || op.as_ref().is::<StableReduceWindowSumOp>()
    {
        let maximum = op.as_ref().is::<StableReduceWindowMaximumOp>();
        let options = if let Some(pool) = op.as_ref().downcast_ref::<StableReduceWindowMaximumOp>()
        {
            pool.get_attr_stable_max_window(ctx).unwrap().options()
        } else {
            op.as_ref()
                .downcast_ref::<StableReduceWindowSumOp>()
                .unwrap()
                .get_attr_stable_sum_window(ctx)
                .unwrap()
                .options()
        };
        let source_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let element_type = stablehlo_element_type(emission.result_type.dtype)?;
        let initializer = floating_initializer(emission.result_type.dtype, maximum)?;
        let auxiliary = emission.auxiliary();
        writeln!(
            emission.body,
            "    {auxiliary} = stablehlo.constant dense<{initializer}> : tensor<{element_type}>"
        )
        .unwrap();
        let window = format!("1, {}, {}, 1", options.window[0], options.window[1]);
        let strides = format!("1, {}, {}, 1", options.strides[0], options.strides[1]);
        let padding = format!(
            "[0, 0], [{}, {}], [{}, {}], [0, 0]",
            options.padding[0][0],
            options.padding[0][1],
            options.padding[1][0],
            options.padding[1][1]
        );
        let reducer = if maximum { "maximum" } else { "add" };
        writeln!(emission.body, "    {} = \"stablehlo.reduce_window\"({}, {auxiliary}) <{{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[{padding}]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: {window}>, window_strides = array<i64: {strides}>}}> ({{", emission.name, emission.operands[0]).unwrap();
        writeln!(
            emission.body,
            "    ^bb0(%window_lhs: tensor<{element_type}>, %window_rhs: tensor<{element_type}>):"
        )
        .unwrap();
        writeln!(emission.body, "      %window_result = stablehlo.{reducer} %window_lhs, %window_rhs : tensor<{element_type}>").unwrap();
        writeln!(
            emission.body,
            "      stablehlo.return %window_result : tensor<{element_type}>"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    }}) : ({source_type}, tensor<{element_type}>) -> {}",
            emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableReduceSumOp>() || op.as_ref().is::<StableReduceMaximumOp>() {
        let maximum = op.as_ref().is::<StableReduceMaximumOp>();
        let axes = if let Some(reduce) = op.as_ref().downcast_ref::<StableReduceSumOp>() {
            reduce.get_attr_stable_sum_axes(ctx).unwrap().values()
        } else {
            op.as_ref()
                .downcast_ref::<StableReduceMaximumOp>()
                .unwrap()
                .get_attr_stable_maximum_axes(ctx)
                .unwrap()
                .values()
        };
        let source_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let element_type = stablehlo_element_type(emission.result_type.dtype)?;
        let initializer = match (maximum, emission.result_type.dtype) {
            (false, DType::I32) => "0",
            (true, DType::I32) => "-2147483648",
            (maximum, dtype) => {
                return Ok(emit_float_reduction(
                    emission,
                    &source_type,
                    element_type,
                    &axes,
                    maximum,
                    &floating_initializer(dtype, maximum)?,
                ));
            }
        };
        let auxiliary = emission.auxiliary();
        writeln!(
            emission.body,
            "    {auxiliary} = stablehlo.constant dense<{initializer}> : tensor<{element_type}>"
        )
        .unwrap();
        let axes = axes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let reducer = if maximum { "maximum" } else { "add" };
        writeln!(emission.body, "    {} = stablehlo.reduce({} init: {auxiliary}) applies stablehlo.{reducer} across dimensions = [{axes}] : ({source_type}, tensor<{element_type}>) -> {}", emission.name, emission.operands[0], emission.ty).unwrap();
    } else {
        return Ok(false);
    }
    Ok(true)
}

fn emit_float_reduction(
    emission: &mut OperationEmission<'_>,
    source_type: &str,
    element_type: &str,
    axes: &[usize],
    maximum: bool,
    initializer: &str,
) -> bool {
    let auxiliary = emission.auxiliary();
    writeln!(
        emission.body,
        "    {auxiliary} = stablehlo.constant dense<{initializer}> : tensor<{element_type}>"
    )
    .unwrap();
    let axes = axes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let reducer = if maximum { "maximum" } else { "add" };
    writeln!(emission.body, "    {} = stablehlo.reduce({} init: {auxiliary}) applies stablehlo.{reducer} across dimensions = [{axes}] : ({source_type}, tensor<{element_type}>) -> {}", emission.name, emission.operands[0], emission.ty).unwrap();
    true
}
