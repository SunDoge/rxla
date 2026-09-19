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
    if let Some(gradient) = op
        .as_ref()
        .downcast_ref::<StableSelectAndScatterMaximumOp>()
    {
        emit_max_pool_gradient(emission, gradient)?;
    } else if op.as_ref().is::<StableReduceWindowMaximumOp>()
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
        if !maximum && matches!(emission.result_type.dtype, DType::F16 | DType::BF16) {
            emit_f32_accumulating_sum(emission, &axes)?;
            return Ok(true);
        }
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

fn emit_max_pool_gradient(
    emission: &mut OperationEmission<'_>,
    gradient: &StableSelectAndScatterMaximumOp,
) -> Result<()> {
    let options = gradient
        .get_attr_stable_max_gradient_window(emission.ctx)
        .unwrap()
        .options();
    let operation = emission.operation.deref(emission.ctx);
    let input_type = value_type(emission.ctx, operation.get_operand(0));
    let gradient_type = stablehlo_type(&value_type(emission.ctx, operation.get_operand(1)))?;
    let element_type = stablehlo_element_type(input_type.dtype)?;
    let negative_infinity = floating_initializer(input_type.dtype, true)?;
    let zero = floating_initializer(input_type.dtype, false)?;
    let padding = [[0, 0], options.padding[0], options.padding[1], [0, 0]];
    let padded_type = TensorType {
        dims: input_type
            .dims
            .iter()
            .zip(padding)
            .map(|(&dimension, [low, high])| {
                if dimension < 0 {
                    dimension
                } else {
                    dimension + low + high
                }
            })
            .collect(),
        ..input_type.clone()
    };
    let padded_type = stablehlo_type(&padded_type)?;
    let pad_value = emission.auxiliary();
    let padded = emission.auxiliary();
    let scatter_init = emission.auxiliary();
    let scattered = emission.auxiliary();
    writeln!(
        emission.body,
        "    {pad_value} = stablehlo.constant dense<{negative_infinity}> : tensor<{element_type}>"
    )
    .unwrap();
    writeln!(emission.body, "    {padded} = stablehlo.pad {}, {pad_value}, low = [{}, {}, {}, {}], high = [{}, {}, {}, {}], interior = [0, 0, 0, 0] : ({}, tensor<{element_type}>) -> {padded_type}", emission.operands[0], padding[0][0], padding[1][0], padding[2][0], padding[3][0], padding[0][1], padding[1][1], padding[2][1], padding[3][1], stablehlo_type(&input_type)?).unwrap();
    writeln!(
        emission.body,
        "    {scatter_init} = stablehlo.constant dense<{zero}> : tensor<{element_type}>"
    )
    .unwrap();
    let window = format!("1, {}, {}, 1", options.window[0], options.window[1]);
    let strides = format!("1, {}, {}, 1", options.strides[0], options.strides[1]);
    writeln!(emission.body, "    {scattered} = \"stablehlo.select_and_scatter\"({padded}, {}, {scatter_init}) <{{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: {window}>, window_strides = array<i64: {strides}>}}> ({{", emission.operands[1]).unwrap();
    writeln!(
        emission.body,
        "    ^bb0(%select_lhs: tensor<{element_type}>, %select_rhs: tensor<{element_type}>):"
    )
    .unwrap();
    writeln!(emission.body, "      %selected = stablehlo.compare GE, %select_lhs, %select_rhs, FLOAT : (tensor<{element_type}>, tensor<{element_type}>) -> tensor<i1>").unwrap();
    writeln!(
        emission.body,
        "      stablehlo.return %selected : tensor<i1>"
    )
    .unwrap();
    writeln!(emission.body, "    }}, {{").unwrap();
    writeln!(
        emission.body,
        "    ^bb0(%scatter_lhs: tensor<{element_type}>, %scatter_rhs: tensor<{element_type}>):"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      %summed = stablehlo.add %scatter_lhs, %scatter_rhs : tensor<{element_type}>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "      stablehlo.return %summed : tensor<{element_type}>"
    )
    .unwrap();
    writeln!(
        emission.body,
        "    }}) : ({padded_type}, {gradient_type}, tensor<{element_type}>) -> {padded_type}"
    )
    .unwrap();

    // Negative edge padding crops the temporary back to the unpadded input
    // shape and remains valid when only the leading batch dimension is dynamic.
    writeln!(emission.body, "    {} = stablehlo.pad {scattered}, {scatter_init}, low = [{}, {}, {}, {}], high = [{}, {}, {}, {}], interior = [0, 0, 0, 0] : ({padded_type}, tensor<{element_type}>) -> {}", emission.name, -padding[0][0], -padding[1][0], -padding[2][0], -padding[3][0], -padding[0][1], -padding[1][1], -padding[2][1], -padding[3][1], emission.ty).unwrap();
    Ok(())
}

fn emit_f32_accumulating_sum(emission: &mut OperationEmission<'_>, axes: &[usize]) -> Result<()> {
    let operation = emission.operation.deref(emission.ctx);
    let source = value_type(emission.ctx, operation.get_operand(0));
    let source_f32 = stablehlo_type(&TensorType {
        dtype: DType::F32,
        ..source
    })?;
    let result_f32 = stablehlo_type(&TensorType {
        dtype: DType::F32,
        ..emission.result_type.clone()
    })?;
    let source_low = stablehlo_type(&value_type(emission.ctx, operation.get_operand(0)))?;
    let converted = emission.auxiliary();
    let initializer = emission.auxiliary();
    let reduced = emission.auxiliary();
    writeln!(
        emission.body,
        "    {converted} = stablehlo.convert {} : ({source_low}) -> {source_f32}",
        emission.operands[0]
    )
    .unwrap();
    writeln!(
        emission.body,
        "    {initializer} = stablehlo.constant dense<0.000000e+00> : tensor<f32>"
    )
    .unwrap();
    let axes = axes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(emission.body, "    {reduced} = stablehlo.reduce({converted} init: {initializer}) applies stablehlo.add across dimensions = [{axes}] : ({source_f32}, tensor<f32>) -> {result_f32}").unwrap();
    writeln!(
        emission.body,
        "    {} = stablehlo.convert {reduced} : ({result_f32}) -> {}",
        emission.name, emission.ty
    )
    .unwrap();
    Ok(())
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
