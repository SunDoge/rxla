//! Pliron dialect conversion from framework semantics to backend semantics.

use super::{super::*, dialect::*};
use pliron::{
    context::Ptr,
    irbuild::{
        cloning::{IrMapping, clone_region_into},
        dialect_conversion::{
            DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
        },
        inserter::Inserter,
        rewriter::Rewriter,
    },
    result::Result as PlironResult,
};

pub(super) fn lower_to_stablehlo(
    ctx: &mut Context,
    module: Ptr<Operation>,
    target: LoweringTarget,
) -> PlironResult<()> {
    apply_dialect_conversion(ctx, &mut RxlaToStableHlo { target }, module)?;
    Ok(())
}

struct RxlaToStableHlo {
    target: LoweringTarget,
}

impl DialectConversion for RxlaToStableHlo {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        supports_source_operation(Operation::get_op_dyn(op, ctx).as_ref())
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _: &OperandsInfo,
    ) -> PlironResult<()> {
        let source = Operation::get_op_dyn(op, ctx);
        let (result_types, operands) = {
            let operation = op.deref(ctx);
            (
                operation.result_types().collect::<Vec<_>>(),
                operation.operands().collect::<Vec<_>>(),
            )
        };
        if source.as_ref().is::<IfOp>() {
            let target = <StableIfOp as PlironOp>::from_operation(Operation::new(
                ctx,
                StableIfOp::get_concrete_op_info(),
                result_types,
                operands[..1].to_vec(),
                vec![],
                2,
            ));
            let mut mapping = IrMapping::new();
            for region_index in 0..2 {
                let source_region = op.deref(ctx).get_region(region_index);
                // Seed captures. Cloning later overwrites mappings for values
                // defined inside the branch, while values defined outside keep
                // their already-lowered identity.
                for block in source_region.deref(ctx).iter(ctx) {
                    for nested in block.deref(ctx).iter(ctx) {
                        for value in nested.deref(ctx).operands() {
                            mapping.map_value(value, value);
                        }
                    }
                }
                let target_region = target.get_operation().deref(ctx).get_region(region_index);
                clone_region_into(source_region, target_region, ctx, rewriter, &mut mapping);
            }
            rewriter.insert_operation(ctx, target.get_operation());
            rewriter.replace_operation(ctx, op, target.get_operation());
            return Ok(());
        }
        macro_rules! replace {
            ($target:ty) => {{
                let target = <$target as PlironOp>::from_operation(Operation::new(
                    ctx,
                    <$target>::get_concrete_op_info(),
                    result_types,
                    operands,
                    vec![],
                    0,
                ));
                copy_sharding(ctx, op, target.get_operation());
                rewriter.insert_operation(ctx, target.get_operation());
                rewriter.replace_operation(ctx, op, target.get_operation());
                return Ok(());
            }};
        }
        macro_rules! replace_attr {
            ($target:ty, $setter:ident, $value:expr) => {{
                let target = <$target as PlironOp>::from_operation(Operation::new(
                    ctx,
                    <$target>::get_concrete_op_info(),
                    result_types,
                    operands,
                    vec![],
                    0,
                ));
                target.$setter(ctx, $value);
                copy_sharding(ctx, op, target.get_operation());
                rewriter.insert_operation(ctx, target.get_operation());
                rewriter.replace_operation(ctx, op, target.get_operation());
                return Ok(());
            }};
        }
        if let Some(value) = source.as_ref().downcast_ref::<ParameterOp>() {
            replace_attr!(
                ArgumentOp,
                set_attr_abi_number,
                (*value.get_attr_number(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<StateInputOp>() {
            replace_attr!(
                ArgumentOp,
                set_attr_abi_number,
                (*value.get_attr_state_number(ctx).unwrap()).clone()
            );
        }
        if source.as_ref().is::<StateReadOp>() || source.as_ref().is::<StateWriteOp>() {
            replace!(StableReshapeOp);
        }
        if let Some(value) = source.as_ref().downcast_ref::<ConstantOp>() {
            replace_attr!(
                StableConstantOp,
                set_attr_stable_value,
                (*value.get_attr_value(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<IotaOp>() {
            replace_attr!(
                StableIotaOp,
                set_attr_stable_iota_axis,
                (*value.get_attr_iota_axis(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<IntegerBinaryOp>() {
            replace_attr!(
                StableIntegerBinaryOp,
                set_attr_stable_integer_binary,
                (*value.get_attr_integer_binary(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<CompareMaskOp>() {
            replace_attr!(
                StableCompareMaskOp,
                set_attr_stable_comparison,
                (*value.get_attr_comparison(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<TakeOp>() {
            replace_attr!(
                StableTakeOp,
                set_attr_stable_take_axis,
                (*value.get_attr_take_axis(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<TakeAlongAxisOp>() {
            replace_attr!(
                StableTakeAlongAxisOp,
                set_attr_stable_take_along_axis,
                (*value.get_attr_take_along_axis(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<GatherGradientOp>() {
            replace_attr!(
                StableGatherGradientOp,
                set_attr_stable_gather_gradient,
                (*value.get_attr_gather_gradient(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<CumsumOp>() {
            replace_attr!(
                StableCumsumOp,
                set_attr_stable_cumsum_axis,
                (*value.get_attr_axis(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<BroadcastOp>() {
            replace_attr!(
                StableBroadcastOp,
                set_attr_stable_broadcast_axes,
                (*value.get_attr_broadcast_axes(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<TransposeOp>() {
            replace_attr!(
                StableTransposeOp,
                set_attr_stable_permutation,
                (*value.get_attr_permutation(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<ReverseOp>() {
            replace_attr!(
                StableReverseOp,
                set_attr_stable_reverse_axes,
                (*value.get_attr_reverse_axes(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<SliceOp>() {
            replace_attr!(
                StableSliceOp,
                set_attr_stable_slice_spec,
                (*value.get_attr_spec(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<SliceGradientOp>() {
            replace_attr!(
                StableSliceGradientOp,
                set_attr_stable_gradient_spec,
                (*value.get_attr_gradient_spec(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<PadOp>() {
            replace_attr!(
                StablePadOp,
                set_attr_stable_padding,
                (*value.get_attr_padding(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<ConcatenateOp>() {
            replace_attr!(
                StableConcatenateOp,
                set_attr_stable_concatenate_axis,
                (*value.get_attr_concatenate_axis(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<MatmulOp>() {
            replace_attr!(
                StableDotGeneralOp,
                set_attr_stable_batch_rank,
                (*value.get_attr_batch_rank(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<AttentionOp>() {
            let [query, key, value_operand, rest @ ..] = operands.as_slice() else {
                unreachable!("verified attention operand count")
            };
            let mask = match rest {
                [] => None,
                [mask] => Some(*mask),
                _ => unreachable!("verified attention operand count"),
            };
            let source_result_type = {
                let handle = op.deref(ctx).get_result(0).get_type(ctx);
                let type_ref = handle.deref(ctx);
                let ranked = type_ref.downcast_ref::<RankedTensorType>().unwrap();
                TensorType {
                    dims: ranked.shape.values(),
                    dtype: ranked.element.value(),
                }
            };
            if matches!(
                self.target,
                LoweringTarget::CudaF16Attention
                    | LoweringTarget::CudaF16Compute
                    | LoweringTarget::CudaBf16Compute
            ) && mask.is_none()
                && source_result_type.dtype == DType::F32
                && source_result_type.dims.len() == 4
                && source_result_type.dims.iter().all(|&d| d > 0)
                && source_result_type.dims[2] >= 512
                && {
                    let key_handle = key.get_type(ctx);
                    let key_ref = key_handle.deref(ctx);
                    key_ref
                        .downcast_ref::<RankedTensorType>()
                        .is_some_and(|ty| ty.shape.values()[2] == source_result_type.dims[2])
                }
                && source_result_type.dims[3] % 8 == 0
                && source_result_type.dims[3] <= 256
            {
                let target = <StableCudnnAttentionOp as PlironOp>::from_operation(Operation::new(
                    ctx,
                    StableCudnnAttentionOp::get_concrete_op_info(),
                    result_types,
                    operands,
                    vec![],
                    0,
                ));
                target.set_attr_cudnn_attention_scale(
                    ctx,
                    (*value.get_attr_attention_scale(ctx).unwrap()).clone(),
                );
                copy_sharding(ctx, op, target.get_operation());
                rewriter.insert_operation(ctx, target.get_operation());
                rewriter.replace_operation(ctx, op, target.get_operation());
                return Ok(());
            }
            let tensor_type = |value: Value| {
                let handle = value.get_type(ctx);
                let type_ref = handle.deref(ctx);
                let ranked = type_ref.downcast_ref::<RankedTensorType>().unwrap();
                TensorType {
                    dims: ranked.shape.values(),
                    dtype: ranked.element.value(),
                }
            };
            let type_handle = |ctx: &Context, ty: &TensorType| -> TypeHandle {
                RankedTensorType::get(
                    ctx,
                    ShapeAttr::new(&ty.dims),
                    ElementTypeAttr::new(ty.dtype),
                )
                .into()
            };
            macro_rules! insert {
                ($kind:ty, $ty:expr, $operands:expr) => {{
                    let target = <$kind as PlironOp>::from_operation(Operation::new(
                        ctx,
                        <$kind>::get_concrete_op_info(),
                        vec![type_handle(ctx, &$ty)],
                        $operands,
                        vec![],
                        0,
                    ));
                    let result = target.get_result(ctx);
                    rewriter.insert_operation(ctx, target.get_operation());
                    (target, result)
                }};
            }

            let query_type = tensor_type(*query);
            let key_type = tensor_type(*key);
            let value_type = tensor_type(*value_operand);
            let rank = query_type.dims.len();
            let mut output_type = query_type.clone();
            output_type.dims[rank - 1] = value_type.dims[rank - 1];
            let batch_rank = rank - 2;
            let mut key_permutation = (0..rank).collect::<Vec<_>>();
            key_permutation.swap(rank - 2, rank - 1);
            let mut transposed_key_type = key_type.clone();
            transposed_key_type.dims.swap(rank - 2, rank - 1);
            let (transpose, transposed_key) =
                insert!(StableTransposeOp, transposed_key_type, vec![*key]);
            transpose.set_attr_stable_permutation(ctx, AxesAttr::new(&key_permutation));

            let mut score_type = query_type.clone();
            score_type.dims[rank - 1] = key_type.dims[rank - 2];
            let (first_dot, scores) = insert!(
                StableDotGeneralOp,
                score_type.clone(),
                vec![*query, transposed_key]
            );
            first_dot.set_attr_stable_batch_rank(ctx, BatchRankAttr::new(batch_rank));

            let scalar_type = TensorType {
                dims: vec![],
                dtype: DType::F32,
            };
            let (scale_constant, scale_value) = insert!(StableConstantOp, scalar_type, vec![]);
            scale_constant.set_attr_stable_value(
                ctx,
                BytesAttr::new(
                    value
                        .get_attr_attention_scale(ctx)
                        .unwrap()
                        .value()
                        .to_le_bytes()
                        .to_vec(),
                ),
            );
            let (scale_broadcast, scale_value) =
                insert!(StableBroadcastOp, score_type.clone(), vec![scale_value]);
            scale_broadcast.set_attr_stable_broadcast_axes(ctx, AxesAttr::new(&[]));
            let (_, mut scores) = insert!(
                StableMultiplyOp,
                score_type.clone(),
                vec![scores, scale_value]
            );
            if let Some(mask) = mask {
                scores = insert!(StableAddOp, score_type.clone(), vec![scores, mask]).1;
            }

            let axis = rank - 1;
            let mut reduced_type = score_type.clone();
            reduced_type.dims.remove(axis);
            let mut keepdim_type = score_type.clone();
            keepdim_type.dims[axis] = 1;
            let (maximum, maximum_value) =
                insert!(StableReduceMaximumOp, reduced_type.clone(), vec![scores]);
            maximum.set_attr_stable_maximum_axes(ctx, AxesAttr::new(&[axis]));
            let (_, maximum_value) =
                insert!(StableReshapeOp, keepdim_type.clone(), vec![maximum_value]);
            let (maximum_broadcast, maximum_value) =
                insert!(StableBroadcastOp, score_type.clone(), vec![maximum_value]);
            maximum_broadcast
                .set_attr_stable_broadcast_axes(ctx, AxesAttr::new(&(0..rank).collect::<Vec<_>>()));
            let (_, centered) = insert!(
                StableSubtractOp,
                score_type.clone(),
                vec![scores, maximum_value]
            );
            let (_, exponentials) = insert!(StableExpOp, score_type.clone(), vec![centered]);
            let (sum, denominator) = insert!(StableReduceSumOp, reduced_type, vec![exponentials]);
            sum.set_attr_stable_sum_axes(ctx, AxesAttr::new(&[axis]));
            let (_, denominator) = insert!(StableReshapeOp, keepdim_type, vec![denominator]);
            let (denominator_broadcast, denominator) =
                insert!(StableBroadcastOp, score_type.clone(), vec![denominator]);
            denominator_broadcast
                .set_attr_stable_broadcast_axes(ctx, AxesAttr::new(&(0..rank).collect::<Vec<_>>()));
            let (_, probabilities) =
                insert!(StableDivideOp, score_type, vec![exponentials, denominator]);
            let (output, output_value) = insert!(
                StableDotGeneralOp,
                output_type,
                vec![probabilities, *value_operand]
            );
            output.set_attr_stable_batch_rank(ctx, BatchRankAttr::new(batch_rank));
            copy_sharding(ctx, op, output.get_operation());
            rewriter.replace_operation_with_values(ctx, op, vec![output_value]);
            return Ok(());
        }
        if let Some(value) = source.as_ref().downcast_ref::<Conv2dOp>() {
            replace_attr!(
                StableConvolutionOp,
                set_attr_stable_convolution_config,
                ConvolutionConfigAttr::new(value.get_attr_options(ctx).unwrap().options())
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<Conv2dOihwOp>() {
            replace_attr!(
                StableConvolutionOihwOp,
                set_attr_stable_oihw_convolution_config,
                ConvolutionConfigAttr::new(value.get_attr_oihw_options(ctx).unwrap().options())
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<ConvTranspose2dOp>() {
            replace_attr!(
                StableTransposeConvolutionOp,
                set_attr_stable_transpose_convolution_config,
                (*value.get_attr_transpose_options(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<MaxPool2dOp>() {
            replace_attr!(
                StableReduceWindowMaximumOp,
                set_attr_stable_max_window,
                ReduceWindowConfigAttr::new(
                    value.get_attr_max_pool_options(ctx).unwrap().options()
                )
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<SumPool2dOp>() {
            replace_attr!(
                StableReduceWindowSumOp,
                set_attr_stable_sum_window,
                ReduceWindowConfigAttr::new(
                    value.get_attr_sum_pool_options(ctx).unwrap().options()
                )
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<ReduceSumOp>() {
            replace_attr!(
                StableReduceSumOp,
                set_attr_stable_sum_axes,
                (*value.get_attr_reduction_axes(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<ReduceMaximumOp>() {
            replace_attr!(
                StableReduceMaximumOp,
                set_attr_stable_maximum_axes,
                (*value.get_attr_maximum_axes(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<ArgMaxOp>() {
            replace_attr!(
                StableArgMaxOp,
                set_attr_stable_argmax_axis,
                (*value.get_attr_argmax_axis(ctx).unwrap()).clone()
            );
        }
        if let Some(value) = source.as_ref().downcast_ref::<SortedIndicesOp>() {
            replace_attr!(
                StableSortedIndicesOp,
                set_attr_stable_sort,
                (*value.get_attr_sort(ctx).unwrap()).clone()
            );
        }
        if source.as_ref().is::<StopGradientOp>() {
            rewriter.replace_operation_with_values(ctx, op, operands);
            return Ok(());
        }
        if source.as_ref().is::<YieldOp>() {
            replace!(StableReturnOp);
        }
        if source.as_ref().is::<WithGradientOp>()
            || source.as_ref().is::<WithElementwiseDerivativeOp>()
        {
            rewriter.replace_operation_with_values(ctx, op, vec![operands[0]]);
            return Ok(());
        }
        if source.as_ref().is::<ReshapeOp>() {
            replace!(StableReshapeOp);
        }
        if source.as_ref().is::<IndexToFloatOp>()
            || source.as_ref().is::<Bf16ToFloatOp>()
            || source.as_ref().is::<ConvertOp>()
        {
            replace!(StableConvertOp);
        }
        if source.as_ref().is::<IsFiniteMaskOp>() {
            replace!(StableIsFiniteMaskOp);
        }
        if source.as_ref().is::<SelectOp>() {
            replace!(StableSelectOp);
        }
        if source.as_ref().is::<DynamicSliceOp>() {
            replace!(StableDynamicSliceOp);
        }
        if source.as_ref().is::<DynamicUpdateSliceOp>() {
            replace!(StableDynamicUpdateSliceOp);
        }
        if source.as_ref().is::<AddOp>() {
            replace!(StableAddOp);
        }
        if source.as_ref().is::<SubtractOp>() {
            replace!(StableSubtractOp);
        }
        if source.as_ref().is::<MultiplyOp>() {
            replace!(StableMultiplyOp);
        }
        if source.as_ref().is::<DivideOp>() {
            replace!(StableDivideOp);
        }
        if source.as_ref().is::<MaximumOp>() {
            replace!(StableMaximumOp);
        }
        if source.as_ref().is::<MinimumOp>() {
            replace!(StableMinimumOp);
        }
        if source.as_ref().is::<AbsOp>() {
            replace!(StableAbsOp);
        }
        if source.as_ref().is::<NegateOp>() {
            replace!(StableNegateOp);
        }
        if source.as_ref().is::<ExpOp>() {
            replace!(StableExpOp);
        }
        if source.as_ref().is::<LogOp>() {
            replace!(StableLogOp);
        }
        if source.as_ref().is::<SqrtOp>() {
            replace!(StableSqrtOp);
        }
        if source.as_ref().is::<RsqrtOp>() {
            replace!(StableRsqrtOp);
        }
        if source.as_ref().is::<TanhOp>() {
            replace!(StableTanhOp);
        }
        if source.as_ref().is::<SigmoidOp>() {
            replace!(StableLogisticOp);
        }
        if source.as_ref().is::<FloorOp>() {
            replace!(StableFloorOp);
        }
        if source.as_ref().is::<CeilOp>() {
            replace!(StableCeilOp);
        }
        if source.as_ref().is::<RoundOp>() {
            replace!(StableRoundNearestAfzOp);
        }
        if source.as_ref().is::<RoundTiesEvenOp>() {
            replace!(StableRoundNearestEvenOp);
        }
        if source.as_ref().is::<SinOp>() {
            replace!(StableSineOp);
        }
        if source.as_ref().is::<CosOp>() {
            replace!(StableCosineOp);
        }
        if source.as_ref().is::<Log1pOp>() {
            replace!(StableLogPlusOneOp);
        }
        if source.as_ref().is::<Expm1Op>() {
            replace!(StableExponentialMinusOneOp);
        }
        if source.as_ref().is::<ErfOp>() {
            replace!(ChloErfOp);
        }
        if source.as_ref().is::<ReluOp>() {
            replace!(StableReluOp);
        }
        if source.as_ref().is::<SoftplusOp>() {
            replace!(StableSoftplusOp);
        }
        if source.as_ref().is::<OptimizationBarrierOp>() {
            replace!(StableOptimizationBarrierOp);
        }
        unreachable!("only supported source operations are selected")
    }
}

fn copy_sharding(ctx: &Context, source: Ptr<Operation>, target: Ptr<Operation>) {
    let sharding = source
        .deref(ctx)
        .attributes
        .get::<ShardingAttr>(&sharding_attr_key())
        .cloned();
    if let Some(sharding) = sharding {
        target
            .deref_mut(ctx)
            .attributes
            .set(sharding_attr_key(), sharding);
    }
}

pub(super) fn supports_source_operation(op: &dyn PlironOp) -> bool {
    op.is::<ParameterOp>()
        || op.is::<StateInputOp>()
        || op.is::<StateReadOp>()
        || op.is::<StateWriteOp>()
        || op.is::<IfOp>()
        || op.is::<YieldOp>()
        || op.is::<ConstantOp>()
        || op.is::<IotaOp>()
        || op.is::<IntegerBinaryOp>()
        || op.is::<AttentionOp>()
        || op.is::<ConvertOp>()
        || op.is::<CompareMaskOp>()
        || op.is::<IsFiniteMaskOp>()
        || op.is::<SelectOp>()
        || op.is::<DynamicSliceOp>()
        || op.is::<DynamicUpdateSliceOp>()
        || op.is::<TakeOp>()
        || op.is::<TakeAlongAxisOp>()
        || op.is::<GatherGradientOp>()
        || op.is::<CumsumOp>()
        || op.is::<BroadcastOp>()
        || op.is::<TransposeOp>()
        || op.is::<ReverseOp>()
        || op.is::<SliceOp>()
        || op.is::<SliceGradientOp>()
        || op.is::<PadOp>()
        || op.is::<ConcatenateOp>()
        || op.is::<MatmulOp>()
        || op.is::<Conv2dOp>()
        || op.is::<Conv2dOihwOp>()
        || op.is::<ConvTranspose2dOp>()
        || op.is::<MaxPool2dOp>()
        || op.is::<SumPool2dOp>()
        || op.is::<ReduceSumOp>()
        || op.is::<ReduceMaximumOp>()
        || op.is::<ArgMaxOp>()
        || op.is::<SortedIndicesOp>()
        || op.is::<ReshapeOp>()
        || op.is::<IndexToFloatOp>()
        || op.is::<Bf16ToFloatOp>()
        || op.is::<AddOp>()
        || op.is::<SubtractOp>()
        || op.is::<MultiplyOp>()
        || op.is::<DivideOp>()
        || op.is::<MaximumOp>()
        || op.is::<MinimumOp>()
        || op.is::<AbsOp>()
        || op.is::<NegateOp>()
        || op.is::<ExpOp>()
        || op.is::<LogOp>()
        || op.is::<SqrtOp>()
        || op.is::<RsqrtOp>()
        || op.is::<TanhOp>()
        || op.is::<SigmoidOp>()
        || op.is::<FloorOp>()
        || op.is::<CeilOp>()
        || op.is::<RoundOp>()
        || op.is::<RoundTiesEvenOp>()
        || op.is::<SinOp>()
        || op.is::<CosOp>()
        || op.is::<Log1pOp>()
        || op.is::<Expm1Op>()
        || op.is::<ErfOp>()
        || op.is::<ReluOp>()
        || op.is::<SoftplusOp>()
        || op.is::<StopGradientOp>()
        || op.is::<OptimizationBarrierOp>()
        || op.is::<WithGradientOp>()
        || op.is::<WithElementwiseDerivativeOp>()
}
