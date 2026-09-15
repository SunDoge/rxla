//! Serialization of leaf, shape-only, and primitive nonlinear StableHLO ops.

use super::emission_support::{OperationEmission, stablehlo_literal, stablehlo_type, value_type};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_primitive(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let ctx = emission.ctx;
    let operation_ptr = emission.operation;
    let operation = operation_ptr.deref(ctx);
    let op = Operation::get_op_dyn(operation_ptr, ctx);
    if let Some(constant) = op.as_ref().downcast_ref::<StableConstantOp>() {
        let literal = stablehlo_literal(
            emission.result_type,
            constant.get_attr_stable_value(ctx).unwrap().as_ref(),
        )?;
        writeln!(
            emission.body,
            "    {} = stablehlo.constant {literal} : {}",
            emission.name, emission.ty
        )
        .unwrap();
    } else if let Some(iota) = op.as_ref().downcast_ref::<StableIotaOp>() {
        let axis = iota.get_attr_stable_iota_axis(ctx).unwrap().value();
        writeln!(
            emission.body,
            "    {} = stablehlo.iota dim = {axis} : {}",
            emission.name, emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableReshapeOp>() {
        let source = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        writeln!(
            emission.body,
            "    {} = stablehlo.reshape {} : ({source}) -> {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableConvertOp>() {
        let source = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
        writeln!(
            emission.body,
            "    {} = stablehlo.convert {} : ({source}) -> {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else if op.as_ref().is::<StableReluOp>() {
        let zero = emission.auxiliary();
        let zeros = emission.auxiliary();
        let ty = emission.ty;
        writeln!(
            emission.body,
            "    {zero} = stablehlo.constant dense<0x00000000> : tensor<f32>"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {zeros} = stablehlo.broadcast_in_dim {zero}, dims = [] : (tensor<f32>) -> {ty}"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {} = stablehlo.maximum {}, {zeros} : {ty}",
            emission.name, emission.operands[0]
        )
        .unwrap();
    } else if op.as_ref().is::<StableSoftplusOp>() {
        let zero = emission.auxiliary();
        let zeros = emission.auxiliary();
        let positive = emission.auxiliary();
        let absolute = emission.auxiliary();
        let negative = emission.auxiliary();
        let exponential = emission.auxiliary();
        let correction = emission.auxiliary();
        let ty = emission.ty;
        writeln!(
            emission.body,
            "    {zero} = stablehlo.constant dense<0x00000000> : tensor<f32>"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {zeros} = stablehlo.broadcast_in_dim {zero}, dims = [] : (tensor<f32>) -> {ty}"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {positive} = stablehlo.maximum {}, {zeros} : {ty}",
            emission.operands[0]
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {absolute} = stablehlo.abs {} : {ty}",
            emission.operands[0]
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {negative} = stablehlo.negate {absolute} : {ty}"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {exponential} = stablehlo.exponential {negative} : {ty}"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {correction} = stablehlo.log_plus_one {exponential} : {ty}"
        )
        .unwrap();
        writeln!(
            emission.body,
            "    {} = stablehlo.add {positive}, {correction} : {ty}",
            emission.name
        )
        .unwrap();
    } else if op.as_ref().is::<StableOptimizationBarrierOp>() {
        writeln!(
            emission.body,
            "    {} = stablehlo.optimization_barrier {} : {}",
            emission.name, emission.operands[0], emission.ty
        )
        .unwrap();
    } else {
        return Ok(false);
    }
    Ok(true)
}
