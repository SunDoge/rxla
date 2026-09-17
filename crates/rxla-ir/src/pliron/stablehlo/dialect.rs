//! Minimal typed target IR. XLA remains the authoritative StableHLO verifier.

use super::super::*;

#[pliron_attr(
    name = "stablehlo.convolution_config",
    format = "$bytes",
    verifier = "succ"
)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ConvolutionConfigAttr {
    pub(super) bytes: BytesAttr,
}

impl ConvolutionConfigAttr {
    pub(super) fn new(options: Conv2dOptions) -> Self {
        let values = [
            options.strides[0],
            options.strides[1],
            options.padding[0][0],
            options.padding[0][1],
            options.padding[1][0],
            options.padding[1][1],
            options.dilation[0],
            options.dilation[1],
            options.groups,
        ];
        Self {
            bytes: BytesAttr::new(values.into_iter().flat_map(i64::to_le_bytes).collect()),
        }
    }

    pub(super) fn options(&self) -> Conv2dOptions {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(
            remainder.is_empty() && values.len() == 9,
            "malformed StableHLO convolution configuration"
        );
        let value = |index| i64::from_le_bytes(values[index]);
        Conv2dOptions {
            strides: [value(0), value(1)],
            padding: [[value(2), value(3)], [value(4), value(5)]],
            dilation: [value(6), value(7)],
            groups: value(8),
        }
    }
}

#[pliron_attr(
    name = "stablehlo.reduce_window_config",
    format = "$bytes",
    verifier = "succ"
)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ReduceWindowConfigAttr {
    pub(super) bytes: BytesAttr,
}

impl ReduceWindowConfigAttr {
    pub(super) fn new(options: Pool2dOptions) -> Self {
        let values = [
            options.window[0],
            options.window[1],
            options.strides[0],
            options.strides[1],
            options.padding[0][0],
            options.padding[0][1],
            options.padding[1][0],
            options.padding[1][1],
        ];
        Self {
            bytes: BytesAttr::new(values.into_iter().flat_map(i64::to_le_bytes).collect()),
        }
    }

    pub(super) fn options(&self) -> Pool2dOptions {
        let (values, remainder) = self.bytes.as_ref().as_chunks::<8>();
        assert!(
            remainder.is_empty() && values.len() == 8,
            "malformed StableHLO reduce-window configuration"
        );
        let value = |index| i64::from_le_bytes(values[index]);
        Pool2dOptions {
            window: [value(0), value(1)],
            strides: [value(2), value(3)],
            padding: [[value(4), value(5)], [value(6), value(7)]],
        }
    }
}

macro_rules! target_op {
    ($name:ident, $dialect_name:literal, [$($interfaces:tt)*]) => {
        #[pliron_op(name = $dialect_name, format, interfaces = [$($interfaces)*], verifier = "succ")]
        pub(super) struct $name;
    };
}

#[pliron_op(
    name = "rxla_abi.argument",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (abi_number: StringAttr)
)]
pub(super) struct ArgumentOp;

impl Verify for ArgumentOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(number) = self.get_attr_abi_number(ctx) else {
            return pliron::verify_err_noloc!("lowered argument requires an ABI number");
        };
        if number.as_str().parse::<usize>().is_err() {
            return pliron::verify_err_noloc!(
                "lowered argument ABI number must be a nonnegative integer"
            );
        }
        Ok(())
    }
}

#[pliron_op(
    name = "rxla_abi.export",
    format,
    interfaces = [NResultsInterface<0>],
    verifier = "succ"
)]
pub(super) struct ExportOp;

/// Structured target conditional. Result arity is dynamic and its two regions
/// are serialized by the StableHLO emitter.
#[pliron_op(name = "stablehlo.if", format, interfaces = [OneOpdInterface], verifier = "succ")]
pub(super) struct StableIfOp;

#[pliron_op(
    name = "stablehlo.custom_call",
    format,
    interfaces = [OneResultInterface],
    attributes = (
        stable_custom_call_target: StringAttr,
        stable_custom_call_backend_config: StringAttr,
        stable_custom_call_has_side_effect: StringAttr,
        stable_custom_call_api_version: StringAttr
    ),
    verifier = "succ"
)]
pub(super) struct StableCustomCallOp;

#[pliron_op(
    name = "stablehlo.return",
    format,
    interfaces = [NResultsInterface<0>, IsTerminatorInterface],
    verifier = "succ"
)]
pub(super) struct StableReturnOp;

#[pliron_op(
    name = "rxla_cuda.cudnn_attention",
    format,
    interfaces = [NOpdsInterface<3>, OneResultInterface],
    attributes = (cudnn_attention_scale: AttentionScaleAttr),
    verifier = "succ"
)]
pub(super) struct StableCudnnAttentionOp;

#[pliron_op(
    name = "stablehlo.constant",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (stable_value: BytesAttr),
    verifier = "succ"
)]
pub(super) struct StableConstantOp;

#[pliron_op(
    name = "stablehlo.iota",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (stable_iota_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct StableIotaOp;

#[pliron_op(
    name = "stablehlo.broadcast_in_dim",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_broadcast_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct StableBroadcastOp;

#[pliron_op(
    name = "stablehlo.transpose",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_permutation: AxesAttr),
    verifier = "succ"
)]
pub(super) struct StableTransposeOp;

#[pliron_op(
    name = "stablehlo.reverse",
    format,
    interfaces = [OneOpdInterface, OneResultInterface, SameOperandsType, SameResultsType, SameOperandsAndResultType],
    attributes = (stable_reverse_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct StableReverseOp;

#[pliron_op(
    name = "stablehlo.slice",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_slice_spec: SliceSpecAttr),
    verifier = "succ"
)]
pub(super) struct StableSliceOp;

#[pliron_op(
    name = "stablehlo.slice_gradient",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_gradient_spec: SliceSpecAttr),
    verifier = "succ"
)]
pub(super) struct StableSliceGradientOp;

#[pliron_op(
    name = "stablehlo.pad",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_padding: PaddingAttr),
    verifier = "succ"
)]
pub(super) struct StablePadOp;

#[pliron_op(
    name = "stablehlo.concatenate",
    format,
    interfaces = [OneResultInterface],
    attributes = (stable_concatenate_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct StableConcatenateOp;

target_op!(
    StableDynamicSliceOp,
    "stablehlo.dynamic_slice",
    [OneResultInterface]
);

target_op!(
    StableDynamicUpdateSliceOp,
    "stablehlo.dynamic_update_slice",
    [OneResultInterface]
);

#[pliron_op(
    name = "stablehlo.take",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_take_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct StableTakeOp;

#[pliron_op(
    name = "stablehlo.take_along_axis",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_take_along_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct StableTakeAlongAxisOp;

#[pliron_op(
    name = "stablehlo.gather_gradient",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_gather_gradient: GatherGradientAttr),
    verifier = "succ"
)]
pub(super) struct StableGatherGradientOp;

#[pliron_op(
    name = "stablehlo.cumsum",
    format,
    interfaces = [OneOpdInterface, OneResultInterface, SameOperandsType, SameResultsType, SameOperandsAndResultType],
    attributes = (stable_cumsum_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct StableCumsumOp;

#[pliron_op(
    name = "stablehlo.dot_general",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_batch_rank: BatchRankAttr),
    verifier = "succ"
)]
pub(super) struct StableDotGeneralOp;

#[pliron_op(
    name = "stablehlo.convolution",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_convolution_config: ConvolutionConfigAttr),
    verifier = "succ"
)]
pub(super) struct StableConvolutionOp;

#[pliron_op(
    name = "stablehlo.convolution_oihw",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_oihw_convolution_config: ConvolutionConfigAttr),
    verifier = "succ"
)]
pub(super) struct StableConvolutionOihwOp;

#[pliron_op(
    name = "stablehlo.convolution_kernel_gradient",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_kernel_gradient_config: ConvolutionConfigAttr),
    verifier = "succ"
)]
pub(super) struct StableConvolutionKernelGradientOp;

#[pliron_op(
    name = "stablehlo.transpose_convolution",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_transpose_convolution_config: ConvTranspose2dOptionsAttr),
    verifier = "succ"
)]
pub(super) struct StableTransposeConvolutionOp;

#[pliron_op(
    name = "stablehlo.reduce_window_maximum",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_max_window: ReduceWindowConfigAttr),
    verifier = "succ"
)]
pub(super) struct StableReduceWindowMaximumOp;

#[pliron_op(
    name = "stablehlo.reduce_window_sum",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_sum_window: ReduceWindowConfigAttr),
    verifier = "succ"
)]
pub(super) struct StableReduceWindowSumOp;

#[pliron_op(
    name = "stablehlo.reduce_sum",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_sum_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct StableReduceSumOp;

#[pliron_op(
    name = "stablehlo.reduce_maximum",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_maximum_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct StableReduceMaximumOp;

#[pliron_op(
    name = "stablehlo.argmax",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_argmax_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct StableArgMaxOp;

#[pliron_op(
    name = "stablehlo.sorted_indices",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (stable_sort: SortAttr),
    verifier = "succ"
)]
pub(super) struct StableSortedIndicesOp;

target_op!(
    StableReshapeOp,
    "stablehlo.reshape",
    [OneOpdInterface, OneResultInterface]
);

target_op!(
    StableConvertOp,
    "stablehlo.convert",
    [OneOpdInterface, OneResultInterface]
);

#[pliron_op(
    name = "stablehlo.integer_binary",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface, SameOperandsType, SameResultsType, SameOperandsAndResultType],
    attributes = (stable_integer_binary: IntegerBinaryAttr),
    verifier = "succ"
)]
pub(super) struct StableIntegerBinaryOp;

#[pliron_op(
    name = "stablehlo.compare_mask",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (stable_comparison: ComparisonAttr),
    verifier = "succ"
)]
pub(super) struct StableCompareMaskOp;

target_op!(
    StableIsFiniteMaskOp,
    "stablehlo.is_finite_mask",
    [OneOpdInterface, OneResultInterface]
);

target_op!(
    StableSelectOp,
    "stablehlo.select",
    [NOpdsInterface<3>, OneResultInterface]
);

macro_rules! same_type_unary {
    ($name:ident, $dialect_name:literal) => {
        target_op!(
            $name,
            $dialect_name,
            [
                OneOpdInterface,
                OneResultInterface,
                SameOperandsType,
                SameResultsType,
                SameOperandsAndResultType
            ]
        );
    };
}
macro_rules! same_type_binary {
    ($name:ident, $dialect_name:literal) => {
        target_op!($name, $dialect_name, [NOpdsInterface<2>, OneResultInterface, SameOperandsType, SameResultsType, SameOperandsAndResultType]);
    };
}

same_type_binary!(StableAddOp, "stablehlo.add");
same_type_binary!(StableSubtractOp, "stablehlo.subtract");
same_type_binary!(StableMultiplyOp, "stablehlo.multiply");
same_type_binary!(StableDivideOp, "stablehlo.divide");
same_type_binary!(StableMaximumOp, "stablehlo.maximum");
same_type_binary!(StableMinimumOp, "stablehlo.minimum");
same_type_unary!(StableAbsOp, "stablehlo.abs");
same_type_unary!(StableNegateOp, "stablehlo.negate");
same_type_unary!(StableExpOp, "stablehlo.exponential");
same_type_unary!(StableLogOp, "stablehlo.log");
same_type_unary!(StableSqrtOp, "stablehlo.sqrt");
same_type_unary!(StableRsqrtOp, "stablehlo.rsqrt");
same_type_unary!(StableTanhOp, "stablehlo.tanh");
same_type_unary!(StableLogisticOp, "stablehlo.logistic");
same_type_unary!(StableFloorOp, "stablehlo.floor");
same_type_unary!(StableCeilOp, "stablehlo.ceil");
same_type_unary!(StableRoundNearestAfzOp, "stablehlo.round_nearest_afz");
same_type_unary!(StableRoundNearestEvenOp, "stablehlo.round_nearest_even");
same_type_unary!(StableSineOp, "stablehlo.sine");
same_type_unary!(StableCosineOp, "stablehlo.cosine");
same_type_unary!(StableLogPlusOneOp, "stablehlo.log_plus_one");
same_type_unary!(
    StableExponentialMinusOneOp,
    "stablehlo.exponential_minus_one"
);
same_type_unary!(StableReluOp, "stablehlo.relu");
same_type_unary!(StableSoftplusOp, "stablehlo.softplus");
same_type_unary!(
    StableOptimizationBarrierOp,
    "stablehlo.optimization_barrier"
);
same_type_unary!(ChloErfOp, "chlo.erf");
