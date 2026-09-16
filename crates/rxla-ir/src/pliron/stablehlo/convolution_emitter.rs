//! Serialization of forward and transposed convolution configurations.

use super::emission_support::{OperationEmission, stablehlo_type, value_type};
use super::{super::*, dialect::*};
use std::fmt::Write;

fn pair(values: [i64; 2]) -> String {
    format!("{}, {}", values[0], values[1])
}

pub(super) fn emit_convolution(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let operation_ptr = emission.operation;
    let operation = operation_ptr.deref(ctx);
    let op = Operation::get_op_dyn(operation_ptr, ctx);
    if let Some(convolution) = op.as_ref().downcast_ref::<StableConvolutionOp>() {
        let options = convolution
            .get_attr_stable_convolution_config(ctx)
            .unwrap()
            .options();
        emit_forward_convolution(emission, options, "[0, 1, i, o]")?;
    } else if let Some(convolution) = op.as_ref().downcast_ref::<StableConvolutionOihwOp>() {
        let options = convolution
            .get_attr_stable_oihw_convolution_config(ctx)
            .unwrap()
            .options();
        emit_forward_convolution(emission, options, "[o, i, 0, 1]")?;
    } else if let Some(convolution) = op
        .as_ref()
        .downcast_ref::<StableConvolutionKernelGradientOp>()
    {
        let options = convolution
            .get_attr_stable_kernel_gradient_config(ctx)
            .unwrap()
            .options();
        emit_kernel_gradient_convolution(emission, options)?;
    } else if let Some(convolution) = op.as_ref().downcast_ref::<StableTransposeConvolutionOp>() {
        let lhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        let rhs = value_type(ctx, operation.get_operand(1));
        let rhs_type = stablehlo_type(&rhs)?;
        let options = convolution
            .get_attr_stable_transpose_convolution_config(ctx)
            .unwrap()
            .options();
        let extent = [
            (rhs.dims[0] - 1) * options.dilation[0],
            (rhs.dims[1] - 1) * options.dilation[1],
        ];
        let padding = format!(
            "[{}, {}], [{}, {}]",
            extent[0] - options.padding[0][0],
            extent[0] - options.padding[0][1] + options.output_padding[0],
            extent[1] - options.padding[1][0],
            extent[1] - options.padding[1][1] + options.output_padding[1]
        );
        writeln!(
            emission.body,
            "    {} = stablehlo.convolution({}, {}) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {{stride = [1, 1], pad = [{padding}], lhs_dilate = [{}], rhs_dilate = [{}], reverse = [true, true]}} {{batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]}} : ({lhs_type}, {rhs_type}) -> {}",
            emission.name,
            emission.operands[0],
            emission.operands[1],
            pair(options.strides),
            pair(options.dilation),
            emission.ty,
        )
        .unwrap();
    } else {
        return Ok(false);
    }
    Ok(true)
}

fn emit_kernel_gradient_convolution(
    emission: &mut OperationEmission<'_>,
    options: Conv2dOptions,
) -> Result<()> {
    let ctx = emission.ctx;
    let operation = emission.operation.deref(ctx);
    let input = value_type(ctx, operation.get_operand(0));
    let output_gradient = value_type(ctx, operation.get_operand(1));
    let kernel = value_type(ctx, operation.get_result(0));
    let lhs_type = stablehlo_type(&input)?;
    let rhs_type = stablehlo_type(&output_gradient)?;
    let mut padding = [[0_i64; 2]; 2];
    for (axis, axis_padding) in padding.iter_mut().enumerate() {
        let output_dilated = (output_gradient.dims[axis + 1] - 1) * options.strides[axis] + 1;
        let kernel_dilated = (kernel.dims[axis] - 1) * options.dilation[axis] + 1;
        axis_padding[0] = options.padding[axis][0];
        axis_padding[1] =
            output_dilated - input.dims[axis + 1] + kernel_dilated - options.padding[axis][0] - 1;
    }
    writeln!(
        emission.body,
        "    {} = stablehlo.convolution({}, {}) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {{stride = [{}], pad = [[{}, {}], [{}, {}]], lhs_dilate = [1, 1], rhs_dilate = [{}], reverse = [false, false]}} {{batch_group_count = {} : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]}} : ({lhs_type}, {rhs_type}) -> {}",
        emission.name,
        emission.operands[0],
        emission.operands[1],
        pair(options.dilation),
        padding[0][0],
        padding[0][1],
        padding[1][0],
        padding[1][1],
        pair(options.strides),
        options.groups,
        emission.ty,
    )
    .unwrap();
    Ok(())
}

fn emit_forward_convolution(
    emission: &mut OperationEmission<'_>,
    options: Conv2dOptions,
    kernel_dimensions: &str,
) -> Result<()> {
    let ctx = emission.ctx;
    let operation = emission.operation.deref(ctx);
    let lhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
    let rhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(1)))?;
    let padding = format!(
        "[{}, {}], [{}, {}]",
        options.padding[0][0], options.padding[0][1], options.padding[1][0], options.padding[1][1]
    );
    writeln!(
            emission.body,
            "    {} = stablehlo.convolution({}, {}) dim_numbers = [b, 0, 1, f]x{kernel_dimensions}->[b, 0, 1, f], window = {{stride = [{}], pad = [{padding}], lhs_dilate = [1, 1], rhs_dilate = [{}], reverse = [false, false]}} {{batch_group_count = 1 : i64, feature_group_count = {} : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]}} : ({lhs_type}, {rhs_type}) -> {}",
            emission.name,
            emission.operands[0],
            emission.operands[1],
            pair(options.strides),
            pair(options.dilation),
            options.groups,
            emission.ty,
        )
        .unwrap();
    Ok(())
}
