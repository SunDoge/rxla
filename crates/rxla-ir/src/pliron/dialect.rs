use super::*;
#[pliron_attr(name = "rxla.shape", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ShapeAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.element_type", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ElementTypeAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_type(
    name = "rxla.tensor",
    format = "`<` $shape `x` $element `>`",
    generate_get = true
)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct RankedTensorType {
    pub(super) shape: ShapeAttr,
    pub(super) element: ElementTypeAttr,
}

impl Verify for RankedTensorType {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        self.shape.verify(ctx)?;
        self.element.verify(ctx)?;
        if self.shape.values().iter().any(|&dimension| dimension < -1) {
            return pliron::verify_err_noloc!(
                "rxla tensor dimensions must be nonnegative or -1 for dynamic"
            );
        }
        if supported_dtype(self.element.value()).is_none() {
            return pliron::verify_err_noloc!("rxla tensor has an unsupported element type");
        }
        Ok(())
    }
}

#[pliron_attr(name = "rxla.conv2d_options", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct Conv2dOptionsAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.conv_transpose2d_options", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ConvTranspose2dOptionsAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.pool2d_options", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct Pool2dOptionsAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.axes", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct AxesAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.axis", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct AxisAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.slice_spec", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct SliceSpecAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.padding", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct PaddingAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.comparison", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ComparisonAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.integer_binary", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct IntegerBinaryAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.batch_rank", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct BatchRankAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.attention_scale", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct AttentionScaleAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.gather_gradient", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct GatherGradientAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.sharding", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ShardingAttr {
    pub(super) bytes: BytesAttr,
}

#[pliron_attr(name = "rxla.sort", format = "$bytes")]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct SortAttr {
    pub(super) bytes: BytesAttr,
}

impl SortAttr {
    pub(super) fn new(axis: usize, descending: bool) -> Self {
        let mut bytes = (axis as u64).to_le_bytes().to_vec();
        bytes.push(u8::from(descending));
        Self {
            bytes: BytesAttr::new(bytes),
        }
    }

    pub(super) fn values(&self) -> (usize, bool) {
        let bytes = self.bytes.as_ref();
        assert_eq!(bytes.len(), 9, "malformed sort attribute");
        let axis = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        (axis, bytes[8] != 0)
    }
}

#[pliron_op(
    name = "rxla.parameter",
    format = "attr($number, $StringAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (number: StringAttr)
)]
pub(super) struct ParameterOp;

/// Stateful reference input before effect discharge. State identity is logical
/// metadata; the result remains an ordinary ranked tensor SSA value.
#[pliron_op(
    name = "rxla.state_input",
    format = "attr($state_number, $StringAttr) ` ` attr($input_state_id, $StringAttr) ` ` attr($state_path, $StringAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (state_number: StringAttr, input_state_id: StringAttr, state_path: StringAttr)
)]
pub(super) struct StateInputOp;

/// Explicit state read. Its operand is the current SSA version and its result
/// is discharged to that value before backend lowering.
#[pliron_op(
    name = "rxla.state_read",
    format = "$0 ` ` attr($read_state_id, $StringAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    attributes = (read_state_id: StringAttr)
)]
pub(super) struct StateReadOp;

/// Explicit state write. The result is the next SSA version of the reference.
#[pliron_op(
    name = "rxla.state_write",
    format = "$0 ` ` attr($write_state_id, $StringAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    attributes = (write_state_id: StringAttr)
)]
pub(super) struct StateWriteOp;

#[pliron_op(
    name = "rxla.constant",
    format = "attr($value, $BytesAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (value: BytesAttr)
)]
pub(super) struct ConstantOp;

#[pliron_op(
    name = "rxla.iota",
    format = "attr($iota_axis, $AxisAttr) ` : ` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (iota_axis: AxisAttr)
)]
pub(super) struct IotaOp;

/// Structured conditional with one I32 scalar predicate, arbitrary result
/// arity, and exactly two single-block regions. Branch results are carried by
/// their `rxla.yield` terminators, never as parent operands: nested values do
/// not dominate their enclosing operation.
#[pliron_op(name = "rxla.if", format, interfaces = [OneOpdInterface])]
pub(super) struct IfOp;

/// Terminates an `rxla.if` branch with its yielded values.
#[pliron_op(
    name = "rxla.yield",
    format,
    interfaces = [NResultsInterface<0>, IsTerminatorInterface],
    verifier = "succ"
)]
pub(super) struct YieldOp;

#[pliron_op(
    name = "rxla.integer_binary",
    format = "$0 `, ` $1 ` ` attr($integer_binary, $IntegerBinaryAttr) ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    attributes = (integer_binary: IntegerBinaryAttr),
    verifier = "succ"
)]
pub(super) struct IntegerBinaryOp;

#[pliron_op(
    name = "rxla.add",
    format = "$0 `, ` $1 ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    verifier = "succ"
)]
pub(super) struct AddOp;

#[pliron_op(
    name = "rxla.multiply",
    format = "$0 `, ` $1 ` : ` type($0)",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    verifier = "succ"
)]
pub(super) struct MultiplyOp;

#[pliron_op(
    name = "rxla.matmul",
    format = "$0 `, ` $1 ` ` attr($batch_rank, $BatchRankAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (batch_rank: BatchRankAttr),
    verifier = "succ"
)]
pub(super) struct MatmulOp;

#[pliron_op(
    name = "rxla.attention",
    format,
    interfaces = [OneResultInterface],
    attributes = (attention_scale: AttentionScaleAttr)
)]
pub(super) struct AttentionOp;

#[pliron_op(
    name = "rxla.conv2d",
    format = "$0 `, ` $1 ` ` attr($options, $Conv2dOptionsAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (options: Conv2dOptionsAttr),
    verifier = "succ"
)]
pub(super) struct Conv2dOp;

#[pliron_op(
    name = "rxla.conv2d_oihw",
    format = "$0 `, ` $1 ` ` attr($oihw_options, $Conv2dOptionsAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (oihw_options: Conv2dOptionsAttr),
    verifier = "succ"
)]
pub(super) struct Conv2dOihwOp;

#[pliron_op(
    name = "rxla.conv2d_kernel_gradient",
    format = "$0 `, ` $1 ` ` attr($kernel_gradient_options, $Conv2dOptionsAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (kernel_gradient_options: Conv2dOptionsAttr),
    verifier = "succ"
)]
pub(super) struct Conv2dKernelGradientOp;

#[pliron_op(
    name = "rxla.conv_transpose2d",
    format = "$0 `, ` $1 ` ` attr($transpose_options, $ConvTranspose2dOptionsAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (transpose_options: ConvTranspose2dOptionsAttr)
)]
pub(super) struct ConvTranspose2dOp;

#[pliron_op(
    name = "rxla.max_pool2d",
    format = "$0 ` ` attr($max_pool_options, $Pool2dOptionsAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (max_pool_options: Pool2dOptionsAttr),
    verifier = "succ"
)]
pub(super) struct MaxPool2dOp;

#[pliron_op(
    name = "rxla.sum_pool2d",
    format = "$0 ` ` attr($sum_pool_options, $Pool2dOptionsAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (sum_pool_options: Pool2dOptionsAttr),
    verifier = "succ"
)]
pub(super) struct SumPool2dOp;

macro_rules! define_same_type_unary_op {
    ($name:ident, $dialect_name:literal) => {
        #[pliron_op(
                                                                    name = $dialect_name,
                                                                    format = "$0 ` : ` type($0)",
                                                            interfaces = [
                                                                OneOpdInterface,
                                                                OneResultInterface,
                                                                SameOperandsType,
                                                                SameResultsType,
                                                                SameOperandsAndResultType
                                                            ],
                                                                    verifier = "succ"
                                                                )]
        pub(super) struct $name;
    };
}

macro_rules! define_same_type_binary_op {
    ($name:ident, $dialect_name:literal) => {
        #[pliron_op(
                                                            name = $dialect_name,
                                                            format = "$0 `, ` $1 ` : ` type($0)",
                                                            interfaces = [
                                                                NOpdsInterface<2>,
                                                                OneResultInterface,
                                                                SameOperandsType,
                                                                SameResultsType,
                                                                SameOperandsAndResultType
                                                            ],
                                                            verifier = "succ"
                                                        )]
        pub(super) struct $name;
    };
}

macro_rules! define_typed_unary_op {
    ($name:ident, $dialect_name:literal) => {
        #[pliron_op(
                                            name = $dialect_name,
                                            format = "$0 ` : ` type($0) ` -> ` type($1)",
                                            interfaces = [OneOpdInterface, OneResultInterface],
                                            verifier = "succ"
                                        )]
        pub(super) struct $name;
    };
}

define_same_type_binary_op!(SubtractOp, "rxla.subtract");
define_same_type_binary_op!(DivideOp, "rxla.divide");
define_same_type_binary_op!(MaximumOp, "rxla.maximum");
define_same_type_binary_op!(MinimumOp, "rxla.minimum");

define_same_type_unary_op!(FloorOp, "rxla.floor");
define_same_type_unary_op!(CeilOp, "rxla.ceil");
define_same_type_unary_op!(RoundOp, "rxla.round");
define_same_type_unary_op!(RoundTiesEvenOp, "rxla.round_ties_even");
define_same_type_unary_op!(SinOp, "rxla.sin");
define_same_type_unary_op!(CosOp, "rxla.cos");
define_same_type_unary_op!(ErfOp, "rxla.erf");
define_same_type_unary_op!(ExpOp, "rxla.exp");
define_same_type_unary_op!(LogOp, "rxla.log");
define_same_type_unary_op!(Log1pOp, "rxla.log1p");
define_same_type_unary_op!(Expm1Op, "rxla.expm1");
define_same_type_unary_op!(AbsOp, "rxla.abs");
define_same_type_unary_op!(NegateOp, "rxla.negate");
define_same_type_unary_op!(SqrtOp, "rxla.sqrt");
define_same_type_unary_op!(RsqrtOp, "rxla.rsqrt");
define_same_type_unary_op!(TanhOp, "rxla.tanh");
define_same_type_unary_op!(ReluOp, "rxla.relu");
define_same_type_unary_op!(SoftplusOp, "rxla.softplus");
define_same_type_unary_op!(SigmoidOp, "rxla.sigmoid");
define_same_type_unary_op!(StopGradientOp, "rxla.stop_gradient");
define_same_type_unary_op!(OptimizationBarrierOp, "rxla.optimization_barrier");

#[pliron_op(
    name = "rxla.with_gradient",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface]
)]
pub(super) struct WithGradientOp;

#[pliron_op(
    name = "rxla.with_elementwise_derivative",
    format,
    interfaces = [OneResultInterface]
)]
pub(super) struct WithElementwiseDerivativeOp;

define_typed_unary_op!(IndexToFloatOp, "rxla.index_to_float");
define_typed_unary_op!(Bf16ToFloatOp, "rxla.bf16_to_float");
define_typed_unary_op!(ConvertOp, "rxla.convert");

#[pliron_op(
    name = "rxla.reshape",
    format = "$0 ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface]
)]
pub(super) struct ReshapeOp;

#[pliron_op(
    name = "rxla.broadcast",
    format = "$0 ` ` attr($broadcast_axes, $AxesAttr) ` : ` type($0)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (broadcast_axes: AxesAttr)
)]
pub(super) struct BroadcastOp;

#[pliron_op(
    name = "rxla.transpose",
    format = "$0 ` ` attr($permutation, $AxesAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (permutation: AxesAttr)
)]
pub(super) struct TransposeOp;

#[pliron_op(
    name = "rxla.reverse",
    format = "$0 ` ` attr($reverse_axes, $AxesAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    attributes = (reverse_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct ReverseOp;

#[pliron_op(
    name = "rxla.cumsum",
    format = "$0 ` ` attr($axis, $AxisAttr) ` : ` type($0)",
    interfaces = [
        OneOpdInterface,
        OneResultInterface,
        SameOperandsType,
        SameResultsType,
        SameOperandsAndResultType
    ],
    attributes = (axis: AxisAttr)
)]
pub(super) struct CumsumOp;

#[pliron_op(
    name = "rxla.slice",
    format = "$0 ` ` attr($spec, $SliceSpecAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (spec: SliceSpecAttr),
    verifier = "succ"
)]
pub(super) struct SliceOp;

#[pliron_op(
    name = "rxla.slice_gradient",
    format = "$0 ` ` attr($gradient_spec, $SliceSpecAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (gradient_spec: SliceSpecAttr)
)]
pub(super) struct SliceGradientOp;

#[pliron_op(
    name = "rxla.pad",
    format = "$0 `, ` $1 ` ` attr($padding, $PaddingAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (padding: PaddingAttr),
    verifier = "succ"
)]
pub(super) struct PadOp;

#[pliron_op(
    name = "rxla.dynamic_slice",
    format,
    interfaces = [OneResultInterface]
)]
pub(super) struct DynamicSliceOp;

#[pliron_op(
    name = "rxla.dynamic_update_slice",
    format,
    interfaces = [OneResultInterface]
)]
pub(super) struct DynamicUpdateSliceOp;

#[pliron_op(
    name = "rxla.take",
    format = "$0 `, ` $1 ` ` attr($take_axis, $AxisAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (take_axis: AxisAttr)
)]
pub(super) struct TakeOp;

#[pliron_op(
    name = "rxla.take_along_axis",
    format = "$0 `, ` $1 ` ` attr($take_along_axis, $AxisAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (take_along_axis: AxisAttr)
)]
pub(super) struct TakeAlongAxisOp;

#[pliron_op(
    name = "rxla.gather_gradient",
    format = "$0 `, ` $1 ` ` attr($gather_gradient, $GatherGradientAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (gather_gradient: GatherGradientAttr)
)]
pub(super) struct GatherGradientOp;

#[pliron_op(
    name = "rxla.concatenate",
    format,
    interfaces = [OneResultInterface],
    attributes = (concatenate_axis: AxisAttr),
    verifier = "succ"
)]
pub(super) struct ConcatenateOp;

#[pliron_op(
    name = "rxla.compare_mask",
    format = "$0 `, ` $1 ` ` attr($comparison, $ComparisonAttr) ` : ` type($0) `, ` type($1) ` -> ` type($2)",
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (comparison: ComparisonAttr),
    verifier = "succ"
)]
pub(super) struct CompareMaskOp;

#[pliron_op(
    name = "rxla.is_finite_mask",
    format = "$0 ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    verifier = "succ"
)]
pub(super) struct IsFiniteMaskOp;

#[pliron_op(
    name = "rxla.select",
    format = "$0 `, ` $1 `, ` $2 ` : ` type($0) `, ` type($1) `, ` type($2) ` -> ` type($3)",
    interfaces = [NOpdsInterface<3>, OneResultInterface],
    verifier = "succ"
)]
pub(super) struct SelectOp;

#[pliron_op(
    name = "rxla.reduce_sum",
    format = "$0 ` ` attr($reduction_axes, $AxesAttr) ` : ` type($0)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (reduction_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct ReduceSumOp;

#[pliron_op(
    name = "rxla.reduce_maximum",
    format = "$0 ` ` attr($maximum_axes, $AxesAttr) ` : ` type($0)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (maximum_axes: AxesAttr),
    verifier = "succ"
)]
pub(super) struct ReduceMaximumOp;

#[pliron_op(
    name = "rxla.argmax",
    format = "$0 ` ` attr($argmax_axis, $AxisAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (argmax_axis: AxisAttr)
)]
pub(super) struct ArgMaxOp;

#[pliron_op(
    name = "rxla.sorted_indices",
    format = "$0 ` ` attr($sort, $SortAttr) ` : ` type($0) ` -> ` type($1)",
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (sort: SortAttr)
)]
pub(super) struct SortedIndicesOp;
