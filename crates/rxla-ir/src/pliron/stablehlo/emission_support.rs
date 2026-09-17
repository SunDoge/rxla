//! Type, reachability, sharding, and opcode helpers for MLIR emission.

use super::{super::*, dialect::*};
use crate::Mesh;
use pliron::context::Ptr;
use std::collections::HashSet;

/// Borrowed state for emitting one already-lowered operation. Additional
/// operation families can live in separate modules without widening the main
/// dispatch function or passing parallel argument lists.
pub(super) struct OperationEmission<'a> {
    pub(super) ctx: &'a Context,
    pub(super) operation: Ptr<Operation>,
    pub(super) operands: &'a [String],
    pub(super) name: &'a str,
    pub(super) ty: &'a str,
    pub(super) result_type: &'a TensorType,
    pub(super) next_auxiliary: &'a mut usize,
    pub(super) body: &'a mut String,
}

impl OperationEmission<'_> {
    pub(super) fn auxiliary(&mut self) -> String {
        let name = format!("%aux{}", self.next_auxiliary);
        *self.next_auxiliary += 1;
        name
    }
}

pub(super) fn sdy_constraint(meshes: &mut Vec<Mesh>, sharding: &Sharding, rank: usize) -> String {
    let mesh = sharding.mesh();
    let mesh_index = meshes
        .iter()
        .position(|candidate| candidate == mesh)
        .unwrap_or_else(|| {
            meshes.push(mesh.clone());
            meshes.len() - 1
        });
    let dimensions = match sharding {
        Sharding::Replicated { .. } => (0..rank).map(|_| "{}".to_owned()).collect::<Vec<_>>(),
        Sharding::Partitioned { spec, .. } => spec
            .axes()
            .iter()
            .map(|axis| match axis {
                Some(axis) => format!("{{\"{}\"}}", escape_mlir_string(axis)),
                None => "{}".to_owned(),
            })
            .collect(),
    };
    format!("<@mesh{mesh_index}, [{}]>", dimensions.join(", "))
}

pub(super) fn sdy_replicated_attr(mesh_index: usize, rank: usize) -> String {
    let dimensions = (0..rank).map(|_| "{}").collect::<Vec<_>>().join(", ");
    format!("#sdy.sharding<@mesh{mesh_index}, [{dimensions}]>")
}

pub(super) fn tensor_rank(ty: &str) -> usize {
    let shape = ty
        .strip_prefix("tensor<")
        .and_then(|ty| ty.strip_suffix('>'))
        .expect("StableHLO emitter constructed a tensor type");
    shape.matches('x').count()
}

pub(super) fn sdy_mesh(index: usize, mesh: &Mesh) -> String {
    let axes = mesh
        .axes()
        .iter()
        .map(|axis| format!("\"{}\"={}", escape_mlir_string(axis.name()), axis.size()))
        .collect::<Vec<_>>()
        .join(", ");
    let stablehlo_axes = mesh
        .axes()
        .iter()
        .map(|axis| {
            format!(
                "{{name = \"{}\", size = {} : i64}}",
                escape_mlir_string(axis.name()),
                axis.size()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "  sdy.mesh @mesh{index} = <[{axes}]> {{stablehlo.mesh = {{axes = [{stablehlo_axes}]}}}}\n"
    )
}

pub(super) fn escape_mlir_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

pub(super) fn reachable_values(ctx: &Context, outputs: &[Value]) -> Result<HashSet<Value>> {
    let mut reachable = HashSet::new();
    let mut worklist = outputs.to_vec();
    for operation in outputs
        .first()
        .and_then(|value| value.defining_op())
        .and_then(|operation| operation.deref(ctx).get_parent_op(ctx))
        .into_iter()
        .flat_map(|module| {
            module
                .deref(ctx)
                .get_region(0)
                .deref(ctx)
                .iter(ctx)
                .flat_map(|block| block.deref(ctx).iter(ctx))
        })
    {
        if let Some(custom) =
            Operation::get_op_dyn(operation, ctx).downcast_ref::<StableCustomCallOp>()
            && custom
                .get_attr_stable_custom_call_has_side_effect(ctx)
                .is_some_and(|value| value.as_str() == "true")
        {
            worklist.extend(operation.deref(ctx).results());
        }
    }
    while let Some(value) = worklist.pop() {
        if !reachable.insert(value) {
            continue;
        }
        let operation = value.defining_op().ok_or(IrError::InvalidValue {
            operation: "walking lowered IR reachability",
        })?;
        let operation = operation.deref(ctx);
        worklist.extend(operation.operands());
        for region_index in 0..operation.num_regions() {
            let region = operation.get_region(region_index);
            for block in region.deref(ctx).iter(ctx) {
                for nested in block.deref(ctx).iter(ctx) {
                    worklist.extend(nested.deref(ctx).operands());
                }
            }
        }
    }
    Ok(reachable)
}

pub(super) fn value_type(ctx: &Context, value: Value) -> TensorType {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    let ranked = ty_ref
        .downcast_ref::<RankedTensorType>()
        .expect("target values retain ranked tensor types");
    TensorType {
        dims: ranked.shape.values(),
        dtype: ranked.element.value(),
        dynamic_bounds: ranked.dynamic_bounds.values(),
    }
}

pub(super) fn stablehlo_literal(ty: &TensorType, bytes: &[u8]) -> Result<String> {
    let (elements, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(IrError::MalformedConstant {
            reason: "StableHLO payload byte length is not divisible by four",
        });
    }
    let values = match ty.dtype {
        DType::F32 => elements
            .iter()
            .map(|bytes| format!("0x{:08X}", u32::from_le_bytes(*bytes)))
            .collect::<Vec<_>>(),
        DType::F16 => elements
            .iter()
            .map(|bytes| {
                let value = f32::from_bits(u32::from_le_bytes(*bytes));
                format!("0x{:04X}", half::f16::from_f32(value).to_bits())
            })
            .collect(),
        DType::BF16 => elements
            .iter()
            .map(|bytes| {
                let value = f32::from_bits(u32::from_le_bytes(*bytes));
                format!("0x{:04X}", half::bf16::from_f32(value).to_bits())
            })
            .collect(),
        DType::I32 => elements
            .iter()
            .map(|bytes| i32::from_le_bytes(*bytes).to_string())
            .collect(),
        dtype => {
            return Err(IrError::UnsupportedDType {
                operation: "StableHLO constant emission",
                dtype,
            });
        }
    };
    let expected = ty.dims.iter().try_fold(1usize, |count, &dimension| {
        usize::try_from(dimension)
            .ok()
            .and_then(|dimension| count.checked_mul(dimension))
    });
    if expected != Some(values.len()) {
        return Err(IrError::MalformedConstant {
            reason: "StableHLO shape does not match its element payload",
        });
    }
    if ty.dims.is_empty()
        || (!values.is_empty() && values.windows(2).all(|pair| pair[0] == pair[1]))
    {
        return Ok(format!("dense<{}>", values[0]));
    }
    Ok(format!(
        "dense<{}>",
        stablehlo_literal_array(&values, &ty.dims)
    ))
}

/// StableHLO's textual `dense` syntax preserves the tensor's nesting. A flat
/// list is accepted only for rank-one tensors; emitting one for rank two or
/// above makes MLIR reject an otherwise well-typed constant.
fn stablehlo_literal_array(values: &[String], dimensions: &[i64]) -> String {
    if dimensions.is_empty() {
        return values[0].clone();
    }
    if dimensions.contains(&0) {
        return "[]".into();
    }
    let width = dimensions[1..]
        .iter()
        .map(|&dimension| usize::try_from(dimension).expect("validated constant shape"))
        .product::<usize>();
    let elements = values
        .chunks(width)
        .map(|chunk| stablehlo_literal_array(chunk, &dimensions[1..]))
        .collect::<Vec<_>>();
    format!("[{}]", elements.join(", "))
}

pub(super) fn stablehlo_type(ty: &TensorType) -> Result<String> {
    let element = stablehlo_element_type(ty.dtype)?;
    let dimensions = ty
        .dims
        .iter()
        .map(|&dimension| {
            if dimension == -1 {
                "?".into()
            } else {
                dimension.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("x");
    let bounds = if ty.dynamic_bounds.is_empty() {
        String::new()
    } else {
        let values = ty
            .dynamic_bounds
            .iter()
            .map(|&bound| {
                if bound == -1 {
                    "?".into()
                } else {
                    bound.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(", #stablehlo.bounds<{values}>")
    };
    Ok(if dimensions.is_empty() {
        format!("tensor<{element}>")
    } else {
        format!("tensor<{dimensions}x{element}{bounds}>")
    })
}

pub(super) fn stablehlo_predicate_type(ty: &TensorType) -> String {
    let dimensions = ty
        .dims
        .iter()
        .map(|&dimension| {
            if dimension == -1 {
                "?".into()
            } else {
                dimension.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("x");
    let bounds = if ty.dynamic_bounds.is_empty() {
        String::new()
    } else {
        let values = ty
            .dynamic_bounds
            .iter()
            .map(|&bound| {
                if bound == -1 {
                    "?".into()
                } else {
                    bound.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(", #stablehlo.bounds<{values}>")
    };
    if dimensions.is_empty() {
        "tensor<i1>".into()
    } else {
        format!("tensor<{dimensions}xi1{bounds}>")
    }
}

pub(super) fn stablehlo_element_type(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::U8 => "ui8",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::BF16 => "bf16",
        dtype => {
            return Err(IrError::UnsupportedDType {
                operation: "StableHLO type emission",
                dtype,
            });
        }
    })
}

pub(super) fn floating_initializer(dtype: DType, negative_infinity: bool) -> Result<String> {
    let value = if negative_infinity {
        f32::NEG_INFINITY
    } else {
        0.0
    };
    Ok(match dtype {
        DType::F32 => format!("0x{:08X}", value.to_bits()),
        DType::F16 => format!("0x{:04X}", half::f16::from_f32(value).to_bits()),
        DType::BF16 => format!("0x{:04X}", half::bf16::from_f32(value).to_bits()),
        dtype => {
            return Err(IrError::UnsupportedDType {
                operation: "StableHLO floating initializer",
                dtype,
            });
        }
    })
}

pub(super) fn opcode(op: &dyn PlironOp) -> Option<&'static str> {
    if op.is::<StableAddOp>() {
        Some("add")
    } else if op.is::<StableSubtractOp>() {
        Some("subtract")
    } else if op.is::<StableMultiplyOp>() {
        Some("multiply")
    } else if op.is::<StableDivideOp>() {
        Some("divide")
    } else if op.is::<StableMaximumOp>() {
        Some("maximum")
    } else if op.is::<StableMinimumOp>() {
        Some("minimum")
    } else if op.is::<StableAbsOp>() {
        Some("abs")
    } else if op.is::<StableNegateOp>() {
        Some("negate")
    } else if op.is::<StableExpOp>() {
        Some("exponential")
    } else if op.is::<StableLogOp>() {
        Some("log")
    } else if op.is::<StableSqrtOp>() {
        Some("sqrt")
    } else if op.is::<StableRsqrtOp>() {
        Some("rsqrt")
    } else if op.is::<StableTanhOp>() {
        Some("tanh")
    } else if op.is::<StableLogisticOp>() {
        Some("logistic")
    } else if op.is::<StableFloorOp>() {
        Some("floor")
    } else if op.is::<StableCeilOp>() {
        Some("ceil")
    } else if op.is::<StableRoundNearestAfzOp>() {
        Some("round_nearest_afz")
    } else if op.is::<StableRoundNearestEvenOp>() {
        Some("round_nearest_even")
    } else if op.is::<StableSineOp>() {
        Some("sine")
    } else if op.is::<StableCosineOp>() {
        Some("cosine")
    } else if op.is::<StableLogPlusOneOp>() {
        Some("log_plus_one")
    } else if op.is::<StableExponentialMinusOneOp>() {
        Some("exponential_minus_one")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multidimensional_constants_use_nested_dense_syntax() {
        let ty = TensorType {
            dims: vec![2, 3],
            dtype: DType::F32,
            dynamic_bounds: vec![],
        };
        let bytes = [1., 2., 3., 4., 5., 6.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            stablehlo_literal(&ty, &bytes).unwrap(),
            "dense<[[0x3F800000, 0x40000000, 0x40400000], [0x40800000, 0x40A00000, 0x40C00000]]>"
        );
    }
}
