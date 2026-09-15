//! Read-only semantic views derived from the canonical Pliron SSA.
//!
//! These values are short-lived analysis results. They are never stored beside
//! the IR, so graph construction has a single source of truth.

use super::program::StableHloProgram;
use super::*;

#[derive(Clone, Debug)]
pub struct SemanticNode {
    pub op: Op,
    pub operands: Vec<SsaId>,
    pub ty: TensorType,
}

/// Backend-independent, handle-free snapshot of one traced program.
#[derive(Clone, Debug)]
pub struct SemanticProgram {
    nodes: Vec<SemanticNode>,
    outputs: Vec<SsaId>,
    preserve_all_inputs: bool,
}

impl SemanticProgram {
    pub fn capture(ir: &ProgramIr, outputs: &[SsaId], preserve_all_inputs: bool) -> Result<Self> {
        for &output in outputs {
            ir.value_type(output)?;
        }
        Ok(Self {
            nodes: ir.semantic_nodes()?,
            outputs: outputs.to_vec(),
            preserve_all_inputs,
        })
    }

    pub fn lower(&self, target: LoweringTarget) -> Result<StableHloProgram> {
        let mut ir = ProgramIr::default();
        for node in &self.nodes {
            let id = ir.append(&node.op, &node.operands, &node.ty)?;
            debug_assert_eq!(id.index(), ir.values.len() - 1);
        }
        ir.stablehlo_program_for(&self.outputs, self.preserve_all_inputs, target)
    }
}

impl ProgramIr {
    pub fn semantic_nodes(&self) -> Result<Vec<SemanticNode>> {
        self.graph.verify("source Pliron")?;
        self.values
            .iter()
            .map(|(id, &value)| {
                Ok(SemanticNode {
                    op: self.graph.semantic_op(value)?,
                    operands: self.operand_ids(id)?,
                    ty: self.graph.value_type(value)?,
                })
            })
            .collect()
    }
}

impl IrGraph {
    fn semantic_op(&self, value: Value) -> Result<Op> {
        let operation = value.defining_op().ok_or(IrError::InvalidValue {
            operation: "projecting Pliron semantics",
        })?;
        let op = Operation::get_op_dyn(operation, &self.ctx);
        let op = op.as_ref();

        macro_rules! plain {
            ($ty:ty, $value:expr) => {
                if op.is::<$ty>() {
                    return Ok($value);
                }
            };
        }
        macro_rules! unary {
            ($ty:ty, $value:expr) => {
                plain!($ty, Op::Unary($value));
            };
        }
        macro_rules! required_attr {
            ($operation:expr, $getter:ident, $attribute:literal) => {
                $operation
                    .$getter(&self.ctx)
                    .ok_or(IrError::MalformedAttribute {
                        attribute: $attribute,
                    })?
            };
        }

        if let Some(value) = op.downcast_ref::<ParameterOp>() {
            let number = value
                .get_attr_number(&self.ctx)
                .ok_or(IrError::MalformedAttribute {
                    attribute: "parameter ABI number",
                })?
                .as_str()
                .parse()
                .map_err(|_| IrError::MalformedAttribute {
                    attribute: "parameter ABI number",
                })?;
            return Ok(Op::Parameter(number));
        }
        if let Some(constant) = op.downcast_ref::<ConstantOp>() {
            let bytes = constant
                .get_attr_value(&self.ctx)
                .ok_or(IrError::MalformedAttribute {
                    attribute: "constant value",
                })?;
            return match self.value_type(value)?.dtype {
                DType::F32 => {
                    let (values, remainder) = bytes.as_ref().as_chunks::<4>();
                    if !remainder.is_empty() {
                        return Err(IrError::MalformedConstant {
                            reason: "F32 payload byte length is not divisible by four",
                        });
                    }
                    Ok(Op::ConstantF32(
                        values
                            .iter()
                            .map(|value| f32::from_le_bytes(*value))
                            .collect::<Vec<_>>()
                            .into(),
                    ))
                }
                DType::I32 => {
                    let (values, remainder) = bytes.as_ref().as_chunks::<4>();
                    if !remainder.is_empty() {
                        return Err(IrError::MalformedConstant {
                            reason: "I32 payload byte length is not divisible by four",
                        });
                    }
                    Ok(Op::ConstantI32(
                        values
                            .iter()
                            .map(|value| i32::from_le_bytes(*value))
                            .collect::<Vec<_>>()
                            .into(),
                    ))
                }
                dtype => Err(IrError::UnsupportedDType {
                    operation: "semantic constant analysis",
                    dtype,
                }),
            };
        }
        if let Some(value) = op.downcast_ref::<IotaOp>() {
            return Ok(Op::Iota {
                axis: required_attr!(value, get_attr_iota_axis, "iota axis").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<IntegerBinaryOp>() {
            return Ok(Op::IntegerBinary(
                required_attr!(value, get_attr_integer_binary, "integer binary operation").value(),
            ));
        }
        if let Some(value) = op.downcast_ref::<MatmulOp>() {
            return Ok(Op::Matmul {
                batch_rank: required_attr!(value, get_attr_batch_rank, "matmul batch rank").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<AttentionOp>() {
            return Ok(Op::Attention {
                scale: required_attr!(value, get_attr_attention_scale, "attention scale").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<Conv2dOp>() {
            return Ok(Op::Conv2d(
                required_attr!(value, get_attr_options, "conv2d options").options(),
            ));
        }
        if let Some(value) = op.downcast_ref::<Conv2dOihwOp>() {
            return Ok(Op::Conv2dOihw(
                required_attr!(value, get_attr_oihw_options, "conv2d OIHW options").options(),
            ));
        }
        if let Some(value) = op.downcast_ref::<ConvTranspose2dOp>() {
            return Ok(Op::ConvTranspose2d(
                required_attr!(value, get_attr_transpose_options, "conv transpose options")
                    .options(),
            ));
        }
        if let Some(value) = op.downcast_ref::<MaxPool2dOp>() {
            return Ok(Op::MaxPool2d(
                required_attr!(value, get_attr_max_pool_options, "max pool options").options(),
            ));
        }
        if let Some(value) = op.downcast_ref::<SumPool2dOp>() {
            return Ok(Op::SumPool2d(
                required_attr!(value, get_attr_sum_pool_options, "sum pool options").options(),
            ));
        }

        plain!(AddOp, Op::Binary(Binary::Add));
        plain!(SubtractOp, Op::Binary(Binary::Sub));
        plain!(MultiplyOp, Op::Binary(Binary::Mul));
        plain!(DivideOp, Op::Binary(Binary::Div));
        plain!(MaximumOp, Op::Binary(Binary::Maximum));
        plain!(MinimumOp, Op::Binary(Binary::Minimum));
        unary!(FloorOp, Unary::Floor);
        unary!(CeilOp, Unary::Ceil);
        unary!(RoundOp, Unary::Round);
        unary!(RoundTiesEvenOp, Unary::RoundTiesEven);
        unary!(SinOp, Unary::Sin);
        unary!(CosOp, Unary::Cos);
        unary!(ErfOp, Unary::Erf);
        unary!(ExpOp, Unary::Exp);
        unary!(LogOp, Unary::Log);
        unary!(Log1pOp, Unary::Log1p);
        unary!(Expm1Op, Unary::Expm1);
        unary!(AbsOp, Unary::Abs);
        unary!(NegateOp, Unary::Neg);
        unary!(SqrtOp, Unary::Sqrt);
        unary!(RsqrtOp, Unary::Rsqrt);
        unary!(TanhOp, Unary::Tanh);
        plain!(ReluOp, Op::Relu);
        plain!(SoftplusOp, Op::Softplus);
        plain!(SigmoidOp, Op::Sigmoid);
        plain!(StopGradientOp, Op::StopGradient);
        plain!(OptimizationBarrierOp, Op::OptimizationBarrier);
        plain!(WithGradientOp, Op::WithGradient);
        plain!(WithElementwiseDerivativeOp, Op::WithElementwiseDerivative);
        plain!(IndexToFloatOp, Op::IndexToFloat);
        plain!(Bf16ToFloatOp, Op::Bf16ToFloat);
        if op.is::<ConvertOp>() {
            return Ok(Op::Convert {
                dtype: self.value_type(value)?.dtype,
            });
        }
        plain!(ReshapeOp, Op::Reshape);

        if let Some(value) = op.downcast_ref::<BroadcastOp>() {
            return Ok(Op::Broadcast {
                axes: required_attr!(value, get_attr_broadcast_axes, "broadcast axes").values(),
            });
        }
        if let Some(value) = op.downcast_ref::<TransposeOp>() {
            return Ok(Op::Transpose {
                permutation: required_attr!(value, get_attr_permutation, "transpose permutation")
                    .values(),
            });
        }
        if let Some(value) = op.downcast_ref::<ReverseOp>() {
            return Ok(Op::Reverse {
                axes: required_attr!(value, get_attr_reverse_axes, "reverse axes").values(),
            });
        }
        if let Some(value) = op.downcast_ref::<CumsumOp>() {
            return Ok(Op::Cumsum {
                axis: required_attr!(value, get_attr_axis, "cumulative sum axis").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<SliceOp>() {
            return Ok(Op::Slice(
                required_attr!(value, get_attr_spec, "slice specification").values(),
            ));
        }
        if let Some(value) = op.downcast_ref::<SliceGradientOp>() {
            return Ok(Op::SliceGradient(
                required_attr!(
                    value,
                    get_attr_gradient_spec,
                    "slice gradient specification"
                )
                .values(),
            ));
        }
        if let Some(value) = op.downcast_ref::<PadOp>() {
            return Ok(Op::Pad(
                required_attr!(value, get_attr_padding, "padding").values(),
            ));
        }
        plain!(DynamicSliceOp, Op::DynamicSlice);
        plain!(DynamicUpdateSliceOp, Op::DynamicUpdateSlice);
        if let Some(value) = op.downcast_ref::<TakeOp>() {
            return Ok(Op::Take {
                axis: required_attr!(value, get_attr_take_axis, "take axis").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<TakeAlongAxisOp>() {
            return Ok(Op::TakeAlongAxis {
                axis: required_attr!(value, get_attr_take_along_axis, "take-along-axis axis")
                    .value(),
            });
        }
        if let Some(value) = op.downcast_ref::<GatherGradientOp>() {
            let (axis, batched) =
                required_attr!(value, get_attr_gather_gradient, "gather gradient options").values();
            return Ok(Op::GatherGradient { axis, batched });
        }
        if let Some(value) = op.downcast_ref::<ConcatenateOp>() {
            return Ok(Op::Concatenate {
                axis: required_attr!(value, get_attr_concatenate_axis, "concatenate axis").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<CompareMaskOp>() {
            return Ok(Op::CompareMask(
                required_attr!(value, get_attr_comparison, "comparison predicate").value(),
            ));
        }
        plain!(IsFiniteMaskOp, Op::IsFiniteMask);
        plain!(SelectOp, Op::Select);
        if let Some(value) = op.downcast_ref::<ReduceSumOp>() {
            return Ok(Op::Reduce {
                kind: Reduction::Sum,
                axes: required_attr!(value, get_attr_reduction_axes, "reduction axes").values(),
            });
        }
        if let Some(value) = op.downcast_ref::<ReduceMaximumOp>() {
            return Ok(Op::Reduce {
                kind: Reduction::Maximum,
                axes: required_attr!(value, get_attr_maximum_axes, "maximum reduction axes")
                    .values(),
            });
        }
        if let Some(value) = op.downcast_ref::<ArgMaxOp>() {
            return Ok(Op::ArgMax {
                axis: required_attr!(value, get_attr_argmax_axis, "argmax axis").value(),
            });
        }
        if let Some(value) = op.downcast_ref::<SortedIndicesOp>() {
            let (axis, descending) = required_attr!(value, get_attr_sort, "sort options").values();
            return Ok(Op::SortedIndices { axis, descending });
        }
        Err(IrError::UnsupportedOperation {
            operation: Operation::get_opid(operation, &self.ctx).to_string(),
        })
    }
}
