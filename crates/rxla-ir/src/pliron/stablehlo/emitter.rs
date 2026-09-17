//! Mechanical serialization of already-lowered target operations.

use super::aggregation_emitter::emit_aggregation;
use super::attention_emitter::emit_attention;
use super::convolution_emitter::emit_convolution;
use super::emission_support::*;
use super::gather_emitter::emit_gather;
use super::indexing_emitter::emit_basic_indexing;
use super::ordered_reduction_emitter::emit_ordered_reduction;
use super::predicate_shape_emitter::emit_predicate_or_shape;
use super::primitive_emitter::emit_primitive;
use super::{super::*, dialect::*};
use crate::Mesh;
use pliron::context::Ptr;
use pliron::region::Region;
use std::fmt::Write;

fn emit_simple_operation(emission: &mut OperationEmission<'_>) -> Result<bool> {
    if emit_attention(emission)?
        || emit_primitive(emission)?
        || emit_predicate_or_shape(emission)?
        || emit_basic_indexing(emission)?
        || emit_gather(emission)?
        || emit_convolution(emission)?
        || emit_aggregation(emission)?
        || emit_ordered_reduction(emission)?
    {
        return Ok(true);
    }
    Ok(false)
}

fn emitted_name(ctx: &Context, names: &HashMap<Value, String>, value: Value) -> Result<String> {
    names
        .get(&value)
        .cloned()
        .or_else(|| {
            Operation::get_op::<ArgumentOp>(value.defining_op()?, ctx).and_then(|argument| {
                argument
                    .get_attr_abi_number(ctx)?
                    .as_str()
                    .parse::<usize>()
                    .ok()
                    .map(|number| format!("%arg{number}"))
            })
        })
        .ok_or(IrError::InvalidValue {
            operation: "emitting StableHLO conditional value",
        })
}

fn emit_structured_if(
    ctx: &Context,
    operation_ptr: Ptr<Operation>,
    operands: &[String],
    names: &mut HashMap<Value, String>,
    next_value: &mut usize,
    next_auxiliary: &mut usize,
    body: &mut String,
) -> Result<()> {
    let operation = operation_ptr.deref(ctx);
    let results = operation.results().collect::<Vec<_>>();
    let result_names = results
        .iter()
        .map(|_| {
            let name = format!("%v{next_value}");
            *next_value += 1;
            name
        })
        .collect::<Vec<_>>();
    let result_types = results
        .iter()
        .map(|result| stablehlo_type(&value_type(ctx, *result)))
        .collect::<Result<Vec<_>>>()?;
    let predicate = if value_type(ctx, operation.get_operand(0)).dtype == DType::I32 {
        let zero = format!("%aux{next_auxiliary}");
        *next_auxiliary += 1;
        let converted = format!("%aux{next_auxiliary}");
        *next_auxiliary += 1;
        writeln!(
            body,
            "    {zero} = stablehlo.constant dense<0> : tensor<i32>"
        )
        .unwrap();
        writeln!(body, "    {converted} = stablehlo.compare NE, {}, {zero}, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>", operands[0]).unwrap();
        converted
    } else {
        operands[0].clone()
    };
    writeln!(
        body,
        "    {} = stablehlo.if {predicate} -> ({}) {{",
        result_names.join(", "),
        result_types.join(", ")
    )
    .unwrap();
    emit_yield_region(
        ctx,
        operation.get_region(0),
        names,
        next_value,
        next_auxiliary,
        body,
    )?;
    writeln!(body, "    }} else {{").unwrap();
    emit_yield_region(
        ctx,
        operation.get_region(1),
        names,
        next_value,
        next_auxiliary,
        body,
    )?;
    writeln!(body, "    }}").unwrap();
    for (result, name) in results.into_iter().zip(result_names) {
        names.insert(result, name);
    }
    Ok(())
}

/// Emit one single-block structured region. Values captured from the parent are
/// resolved from `names`; locally defined SSA values are added as operations are
/// serialized before the `stablehlo.return` terminator.
fn emit_yield_region(
    ctx: &Context,
    region: Ptr<Region>,
    names: &mut HashMap<Value, String>,
    next_value: &mut usize,
    next_auxiliary: &mut usize,
    body: &mut String,
) -> Result<()> {
    let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();
    let [block] = blocks.as_slice() else {
        return Err(IrError::Verification {
            stage: "emitting StableHLO conditional region",
            message: "conditional region must have exactly one block".into(),
        });
    };
    let operations = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
    let Some(&terminator) = operations.last() else {
        return Err(IrError::UnsupportedOperation {
            operation: "empty StableHLO conditional region".into(),
        });
    };
    if !Operation::is_op::<StableReturnOp>(terminator, ctx) {
        return Err(IrError::Verification {
            stage: "emitting StableHLO conditional region",
            message: "conditional region must end with stablehlo.return".into(),
        });
    }
    for operation_ptr in &operations[..operations.len() - 1] {
        let operation = operation_ptr.deref(ctx);
        let operands = operation
            .operands()
            .map(|value| emitted_name(ctx, names, value))
            .collect::<Result<Vec<_>>>()?;
        if Operation::is_op::<StableIfOp>(*operation_ptr, ctx) {
            emit_structured_if(
                ctx,
                *operation_ptr,
                &operands,
                names,
                next_value,
                next_auxiliary,
                body,
            )?;
            continue;
        }
        if operation.get_num_results() != 1 {
            return Err(IrError::UnsupportedOperation {
                operation: "multi-result operation inside a conditional region".into(),
            });
        }
        let result = operation.get_result(0);
        let result_type = value_type(ctx, result);
        let ty = stablehlo_type(&result_type)?;
        let name = format!("%v{next_value}");
        *next_value += 1;
        if !emit_simple_operation(&mut OperationEmission {
            ctx,
            operation: *operation_ptr,
            operands: &operands,
            name: &name,
            ty: &ty,
            result_type: &result_type,
            next_auxiliary,
            body,
        })? {
            let op = Operation::get_op_dyn(*operation_ptr, ctx);
            let opcode = opcode(op.as_ref()).ok_or_else(|| IrError::UnsupportedOperation {
                operation: Operation::get_opid(*operation_ptr, ctx).to_string(),
            })?;
            writeln!(
                body,
                "      {name} = stablehlo.{opcode} {} : {ty}",
                operands.join(", ")
            )
            .unwrap();
        }
        names.insert(result, name);
    }
    let operation = terminator.deref(ctx);
    let values = operation
        .operands()
        .map(|value| emitted_name(ctx, names, value))
        .collect::<Result<Vec<_>>>()?;
    let types = operation
        .operands()
        .map(|value| stablehlo_type(&value_type(ctx, value)))
        .collect::<Result<Vec<_>>>()?;
    writeln!(
        body,
        "      stablehlo.return {} : {}",
        values.join(", "),
        types.join(", ")
    )
    .unwrap();
    Ok(())
}

pub(super) fn emit_module(
    ctx: &Context,
    module: Ptr<Operation>,
    outputs: &[Value],
    preserve_all_inputs: bool,
    target: LoweringTarget,
    boundary_inputs: &[TensorType],
    boundary_outputs: &[TensorType],
) -> Result<String> {
    let module = Operation::get_op::<ModuleOp>(module, ctx).expect("cloned root is a module");
    let mut reachable = reachable_values(ctx, outputs)?;
    if preserve_all_inputs {
        for operation in module.get_body(ctx, 0).deref(ctx).iter(ctx) {
            if Operation::is_op::<ArgumentOp>(operation, ctx) {
                reachable.insert(operation.deref(ctx).get_result(0));
            }
        }
    }
    let mut names = HashMap::new();
    let mut parameters = Vec::new();
    let mut body = String::new();
    let mut next_value = 0usize;
    let mut next_auxiliary = 0usize;
    let mut meshes = Vec::<Mesh>::new();

    let mut arguments = Vec::new();
    for operation_ptr in module.get_body(ctx, 0).deref(ctx).iter(ctx) {
        let Some(argument) = Operation::get_op::<ArgumentOp>(operation_ptr, ctx) else {
            continue;
        };
        let result = operation_ptr.deref(ctx).get_result(0);
        if !reachable.contains(&result) {
            continue;
        }
        let number = argument
            .get_attr_abi_number(ctx)
            .ok_or(IrError::MalformedAttribute {
                attribute: "lowered argument ABI number",
            })?
            .as_str()
            .parse::<usize>()
            .map_err(|_| IrError::MalformedAttribute {
                attribute: "lowered argument ABI number",
            })?;
        arguments.push((number, operation_ptr, result));
    }
    arguments.sort_unstable_by_key(|(number, _, _)| *number);
    if arguments.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(IrError::MalformedAttribute {
            attribute: "unique lowered argument ABI numbers",
        });
    }
    for (_, operation_ptr, result) in arguments {
        let operation = operation_ptr.deref(ctx);
        let argument = format!("%arg{}", parameters.len());
        let result_type = value_type(ctx, result);
        let ty = stablehlo_type(&result_type)?;
        parameters.push((argument.clone(), ty.clone()));
        if let Some(sharding) = operation
            .attributes
            .get::<ShardingAttr>(&sharding_attr_key())
            .map(ShardingAttr::sharding)
        {
            let name = format!("%v{next_value}");
            next_value += 1;
            let constraint = sdy_constraint(&mut meshes, &sharding, result_type.dims.len());
            writeln!(
                body,
                "    {name} = sdy.sharding_constraint {argument} {constraint} : {ty}"
            )
            .unwrap();
            names.insert(result, name);
        } else {
            names.insert(result, argument);
        }
    }

    for operation_ptr in module.get_body(ctx, 0).deref(ctx).iter(ctx) {
        if Operation::is_op::<ExportOp>(operation_ptr, ctx) {
            continue;
        }
        let operation = operation_ptr.deref(ctx);
        let result = operation.get_result(0);
        if !reachable.contains(&result) {
            continue;
        }
        let op = Operation::get_op_dyn(operation_ptr, ctx);
        let result_type = value_type(ctx, result);
        let sharding = operation
            .attributes
            .get::<ShardingAttr>(&sharding_attr_key())
            .map(ShardingAttr::sharding);
        if op.as_ref().is::<ArgumentOp>() {
            continue;
        }
        let operands = operation
            .operands()
            .map(|operand| names.get(&operand).cloned())
            .collect::<Option<Vec<_>>>()
            .ok_or(IrError::InvalidValue {
                operation: "emitting StableHLO operands",
            })?;
        if op.as_ref().is::<StableIfOp>() {
            emit_structured_if(
                ctx,
                operation_ptr,
                &operands,
                &mut names,
                &mut next_value,
                &mut next_auxiliary,
                &mut body,
            )?;
            continue;
        }
        let value_name = format!("%v{next_value}");
        next_value += 1;
        let name = if sharding.is_some() {
            let name = format!("%aux{next_auxiliary}");
            next_auxiliary += 1;
            name
        } else {
            value_name.clone()
        };
        let ty = stablehlo_type(&result_type)?;

        if emit_simple_operation(&mut OperationEmission {
            ctx,
            operation: operation_ptr,
            operands: &operands,
            name: &name,
            ty: &ty,
            result_type: &result_type,
            next_auxiliary: &mut next_auxiliary,
            body: &mut body,
        })? {
        } else if let Some(integer) = op.as_ref().downcast_ref::<StableIntegerBinaryOp>() {
            let opcode = integer
                .get_attr_stable_integer_binary(ctx)
                .unwrap()
                .opcode()
                .replace('-', "_");
            writeln!(
                body,
                "    {name} = stablehlo.{opcode} {}, {} : {ty}",
                operands[0], operands[1]
            )
            .unwrap();
        } else if let Some(dot) = op.as_ref().downcast_ref::<StableDotGeneralOp>() {
            let lhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(0)))?;
            let rhs_type = stablehlo_type(&value_type(ctx, operation.get_operand(1)))?;
            let batch_rank = dot.get_attr_stable_batch_rank(ctx).unwrap().value();
            let batching = if batch_rank == 0 {
                String::new()
            } else {
                let dimensions = (0..batch_rank)
                    .map(|dimension| dimension.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("batching_dims = [{dimensions}] x [{dimensions}], ")
            };
            writeln!(body, "    {name} = stablehlo.dot_general {}, {}, {batching}contracting_dims = [{}] x [{}], precision = [HIGHEST, HIGHEST] : ({lhs_type}, {rhs_type}) -> {ty}", operands[0], operands[1], batch_rank + 1, batch_rank).unwrap();
        } else if let Some(custom) = op.as_ref().downcast_ref::<StableCustomCallOp>() {
            let target = escape_mlir_string(
                custom
                    .get_attr_stable_custom_call_target(ctx)
                    .unwrap()
                    .as_str(),
            );
            let config = escape_mlir_string(
                custom
                    .get_attr_stable_custom_call_backend_config(ctx)
                    .unwrap()
                    .as_str(),
            );
            let side_effect = custom
                .get_attr_stable_custom_call_has_side_effect(ctx)
                .unwrap();
            let api_version = custom.get_attr_stable_custom_call_api_version(ctx).unwrap();
            let operand_types = operation
                .operands()
                .map(|operand| stablehlo_type(&value_type(ctx, operand)))
                .collect::<Result<Vec<_>>>()?;
            writeln!(
                body,
                "    {name} = \"stablehlo.custom_call\"({}) {{call_target_name = \"{target}\", has_side_effect = {}, backend_config = \"{config}\", api_version = {} : i32}} : ({}) -> {ty}",
                operands.join(", "),
                side_effect.as_str(),
                api_version.as_str(),
                operand_types.join(", ")
            )
            .unwrap();
        } else if op.as_ref().is::<ChloErfOp>() {
            writeln!(body, "    {name} = chlo.erf {} : {ty} -> {ty}", operands[0]).unwrap();
        } else {
            let opcode = opcode(op.as_ref()).ok_or_else(|| IrError::UnsupportedOperation {
                operation: Operation::get_opid(operation_ptr, ctx).to_string(),
            })?;
            writeln!(
                body,
                "    {name} = stablehlo.{opcode} {} : {ty}",
                operands.join(", ")
            )
            .unwrap();
        }
        if let Some(sharding) = sharding {
            let constraint = sdy_constraint(&mut meshes, &sharding, result_type.dims.len());
            writeln!(
                body,
                "    {value_name} = sdy.sharding_constraint {name} {constraint} : {ty}"
            )
            .unwrap();
        }
        names.insert(result, value_name);
    }

    let results = outputs
        .iter()
        .map(|output| {
            Ok((
                names.get(output).cloned().ok_or(IrError::InvalidValue {
                    operation: "emitting StableHLO results",
                })?,
                stablehlo_type(&value_type(ctx, *output))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let arguments = parameters
        .iter()
        .map(|(name, ty)| {
            if meshes.is_empty() {
                format!("{name}: {ty}")
            } else {
                format!(
                    "{name}: {ty} {{sdy.sharding = {}}}",
                    sdy_replicated_attr(0, tensor_rank(ty))
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let result_types = results
        .iter()
        .map(|(_, ty)| ty.as_str())
        .collect::<Vec<_>>();
    let boundary_result_types = result_types
        .iter()
        .map(|ty| {
            if meshes.is_empty() {
                (*ty).to_owned()
            } else {
                format!(
                    "{ty} {{sdy.sharding = {}}}",
                    sdy_replicated_attr(0, tensor_rank(ty))
                )
            }
        })
        .collect::<Vec<_>>();
    let result_signature = match boundary_result_types.as_slice() {
        [ty] if !meshes.is_empty() => format!(" -> ({ty})"),
        [ty] => format!(" -> {ty}"),
        types if !types.is_empty() => format!(" -> ({})", types.join(", ")),
        _ => String::new(),
    };
    let module_attributes = meshes
        .first()
        .map(|mesh| {
            format!(
                " attributes {{mhlo.num_partitions = {} : i32, mhlo.num_replicas = 1 : i32}}",
                mesh.device_count().expect("validated mesh size")
            )
        })
        .unwrap_or_default();
    let mesh_declarations = meshes
        .iter()
        .enumerate()
        .map(|(index, mesh)| sdy_mesh(index, mesh))
        .collect::<String>();
    let low_precision = matches!(
        target,
        LoweringTarget::CudaF16Compute | LoweringTarget::CudaBf16Compute
    );
    let function = if low_precision {
        "private @main_compute"
    } else {
        "public @main"
    };
    let mut text = format!(
        "module{module_attributes} {{\n{mesh_declarations}  func.func {function}({arguments}){result_signature} {{\n{body}"
    );
    if results.is_empty() {
        text.push_str("    return\n");
    } else {
        writeln!(
            text,
            "    return {} : {}",
            results
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            result_types.join(", ")
        )
        .unwrap();
    }
    text.push_str("  }\n}\n");
    if low_precision {
        if !meshes.is_empty() {
            return Err(IrError::UnsupportedOperation {
                operation: "low-precision compute with explicit mesh sharding".into(),
            });
        }
        text = low_precision_compute_wrapper(
            text,
            &parameters,
            &results,
            boundary_inputs,
            boundary_outputs,
        )?;
    }
    Ok(text)
}

fn low_precision_compute_wrapper(
    mut inner: String,
    parameters: &[(String, String)],
    results: &[(String, String)],
    boundary_inputs: &[TensorType],
    boundary_outputs: &[TensorType],
) -> Result<String> {
    if parameters.len() != boundary_inputs.len() || results.len() != boundary_outputs.len() {
        return Err(IrError::Verification {
            stage: "emitting low-precision ABI wrapper",
            message: "internal and boundary signatures have different arity".into(),
        });
    }
    inner.truncate(inner.rfind("}\n").expect("emitted module has terminator"));

    let boundary_input_types = boundary_inputs
        .iter()
        .map(stablehlo_type)
        .collect::<Result<Vec<_>>>()?;
    let arguments = boundary_input_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("%arg{index}: {ty}"))
        .collect::<Vec<_>>();
    let inner_input_types = parameters
        .iter()
        .map(|(_, ty)| ty.clone())
        .collect::<Vec<_>>();
    let inner_arguments = parameters
        .iter()
        .enumerate()
        .map(|(index, (_, ty))| {
            if ty != &boundary_input_types[index] {
                format!("%compute_arg{index}")
            } else {
                format!("%arg{index}")
            }
        })
        .collect::<Vec<_>>();
    let output_types = boundary_outputs
        .iter()
        .map(stablehlo_type)
        .collect::<Result<Vec<_>>>()?;
    let inner_output_types = output_types
        .iter()
        .enumerate()
        .map(|(index, _)| results[index].1.clone())
        .collect::<Vec<_>>();
    let result_signature = match output_types.as_slice() {
        [ty] => format!(" -> {ty}"),
        types if !types.is_empty() => format!(" -> ({})", types.join(", ")),
        _ => String::new(),
    };

    writeln!(
        inner,
        "  func.func public @main({}){result_signature} {{",
        arguments.join(", ")
    )
    .unwrap();
    for (index, (_, ty)) in parameters.iter().enumerate() {
        if ty != &boundary_input_types[index] {
            writeln!(
                inner,
                "    %compute_arg{index} = stablehlo.convert %arg{index} : ({}) -> {ty}",
                boundary_input_types[index]
            )
            .unwrap();
        }
    }
    if results.is_empty() {
        writeln!(
            inner,
            "    func.call @main_compute({}) : ({}) -> ()",
            inner_arguments.join(", "),
            inner_input_types.join(", ")
        )
        .unwrap();
        inner.push_str("    return\n");
    } else {
        let call_results = if results.len() == 1 {
            "%compute_result".to_owned()
        } else {
            format!("%compute_result:{}", results.len())
        };
        writeln!(
            inner,
            "    {call_results} = func.call @main_compute({}) : ({}) -> ({})",
            inner_arguments.join(", "),
            inner_input_types.join(", "),
            inner_output_types.join(", ")
        )
        .unwrap();
        let mut returned = Vec::with_capacity(results.len());
        for (index, ty) in output_types.iter().enumerate() {
            let source = if results.len() == 1 {
                "%compute_result".to_owned()
            } else {
                format!("%compute_result#{index}")
            };
            if ty != &inner_output_types[index] {
                let name = format!("%result{index}");
                writeln!(
                    inner,
                    "    {name} = stablehlo.convert {source} : ({}) -> {ty}",
                    inner_output_types[index]
                )
                .unwrap();
                returned.push(name);
            } else {
                returned.push(source);
            }
        }
        writeln!(
            inner,
            "    return {} : {}",
            returned.join(", "),
            output_types.join(", ")
        )
        .unwrap();
    }
    inner.push_str("  }\n}\n");
    Ok(inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_precision_wrapper_preserves_f32_abi() {
        let source = "module {\n  func.func private @main_compute(%arg0: tensor<2xbf16>, %arg1: tensor<i32>) -> tensor<2xbf16> {\n    return %arg0 : tensor<2xbf16>\n  }\n}\n";
        let parameters = vec![
            ("%arg0".into(), "tensor<2xbf16>".into()),
            ("%arg1".into(), "tensor<i32>".into()),
        ];
        let results = vec![("%arg0".into(), "tensor<2xbf16>".into())];
        let inputs = vec![
            TensorType {
                dims: vec![2],
                dtype: DType::F32,
            },
            TensorType {
                dims: vec![],
                dtype: DType::I32,
            },
        ];
        let outputs = vec![TensorType {
            dims: vec![2],
            dtype: DType::F32,
        }];

        let wrapped =
            low_precision_compute_wrapper(source.into(), &parameters, &results, &inputs, &outputs)
                .unwrap();
        assert!(wrapped.contains("private @main_compute(%arg0: tensor<2xbf16>"));
        assert!(wrapped.contains("public @main(%arg0: tensor<2xf32>"));
        assert!(wrapped.contains("stablehlo.convert %arg0"));
        assert!(wrapped.contains("stablehlo.convert %compute_result"));
    }
}
