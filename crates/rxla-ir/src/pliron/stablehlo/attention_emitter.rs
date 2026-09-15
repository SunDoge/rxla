use super::emission_support::{OperationEmission, stablehlo_type, value_type};
use super::{super::*, dialect::*};
use std::fmt::Write;

pub(super) fn emit_attention(emission: &mut OperationEmission<'_>) -> Result<bool> {
    let op = Operation::get_op_dyn(emission.operation, emission.ctx);
    let Some(attention) = op.as_ref().downcast_ref::<StableCudnnAttentionOp>() else {
        return Ok(false);
    };
    let q = value_type(
        emission.ctx,
        emission.operation.deref(emission.ctx).get_operand(0),
    );
    let [batch, heads, queries, depth] = q.dims.as_slice() else {
        unreachable!()
    };
    let keys = value_type(
        emission.ctx,
        emission.operation.deref(emission.ctx).get_operand(1),
    )
    .dims[2];
    let scale = attention
        .get_attr_cudnn_attention_scale(emission.ctx)
        .unwrap()
        .value();
    let operation = emission.operation.deref(emission.ctx);
    let operand_types = (0..3)
        .map(|index| value_type(emission.ctx, operation.get_operand(index)))
        .collect::<Vec<_>>();
    let f16_types = operand_types
        .iter()
        .map(|ty| {
            stablehlo_type(&TensorType {
                dims: ty.dims.clone(),
                dtype: DType::F16,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let output_f16_ty = stablehlo_type(&TensorType {
        dims: emission.result_type.dims.clone(),
        dtype: DType::F16,
    })?;
    let converted = emission
        .operands
        .iter()
        .zip(&operand_types)
        .zip(&f16_types)
        .map(|((operand, source), target)| {
            let name = emission.auxiliary();
            let source = stablehlo_type(source)?;
            writeln!(
                emission.body,
                "    {name} = stablehlo.convert {operand} : ({source}) -> {target}"
            )
            .unwrap();
            Ok(name)
        })
        .collect::<Result<Vec<_>>>()?;
    let config = format!(
        r#"{{"operation_queue_id":"0","cudnn_fmha_backend_config":{{"algorithm":{{"algo_id":"0","math_type":"TENSOR_OP_MATH","tuning_knobs":{{"17":"1","24":"0"}},"is_cudnn_frontend":true,"workspace_size":"0"}},"fmha_scale":{scale},"intermediate_tensor_shape":{{"element_type":"F16","dimensions":["{batch}","{heads}","{queries}","{keys}"],"tuple_shapes":[],"layout":{{"dim_level_types":[],"dim_unique":[],"dim_ordered":[],"minor_to_major":["3","2","1","0"],"tiles":[],"element_size_in_bits":"0","memory_space":"0","index_primitive_type":"PRIMITIVE_TYPE_INVALID","pointer_primitive_type":"PRIMITIVE_TYPE_INVALID","dynamic_shape_metadata_prefix_bytes":"0"}},"is_dynamic_dimension":[false,false,false,false]}},"is_flash_attention":true,"mask_type":"NO_MASK","bmm1_dot_dimension_numbers":{{"lhs_contracting_dimensions":["3"],"rhs_contracting_dimensions":["3"],"lhs_batch_dimensions":["0","1"],"rhs_batch_dimensions":["0","1"]}},"bmm2_dot_dimension_numbers":{{"lhs_contracting_dimensions":["3"],"rhs_contracting_dimensions":["2"],"lhs_batch_dimensions":["0","1"],"rhs_batch_dimensions":["0","1"]}},"dropout_rate":0.0,"seed":42,"sliding_window_length":0,"max_seg_per_batch":1,"is_paged_attention":false}}}}"#
    );
    let escaped = config.replace('\\', "\\\\").replace('"', "\\22");
    let call = emission.auxiliary();
    writeln!(emission.body, "    {call}:2 = stablehlo.custom_call @__cudnn$fmhaSoftmax({}, {}, {}) {{api_version = 2 : i32, backend_config = \"{escaped}\", operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>], result_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]}} : ({}, {}, {}) -> ({output_f16_ty}, tensor<0xui8>)", converted[0], converted[1], converted[2], f16_types[0], f16_types[1], f16_types[2]).unwrap();
    writeln!(
        emission.body,
        "    {} = stablehlo.convert {call}#0 : ({output_f16_ty}) -> {}",
        emission.name, emission.ty
    )
    .unwrap();
    let _ = depth;
    Ok(true)
}
