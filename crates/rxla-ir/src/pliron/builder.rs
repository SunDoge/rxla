use super::*;
impl ProgramIr {
    pub fn append(&mut self, op: &Op, operand_ids: &[SsaId], result: &TensorType) -> Result<SsaId> {
        if let Op::If {
            then_marker,
            then_values,
            else_marker,
            else_values,
            result_types,
        } = op
        {
            let [predicate] = operand_ids else {
                return Err(IrError::InvalidValue {
                    operation: "building a conditional predicate",
                });
            };
            let values = self.append_conditional(
                *predicate,
                *then_marker,
                &then_values
                    .iter()
                    .copied()
                    .map(SsaId::from_index)
                    .collect::<Vec<_>>(),
                *else_marker,
                &else_values
                    .iter()
                    .copied()
                    .map(SsaId::from_index)
                    .collect::<Vec<_>>(),
                result_types,
            )?;
            if result_types.first() != Some(result) {
                return Err(IrError::InvalidValue {
                    operation: "building conditional result metadata",
                });
            }
            return values.first().copied().ok_or(IrError::InvalidValue {
                operation: "building conditional results",
            });
        }
        let operands = operand_ids
            .iter()
            .map(|&id| self.values.get(id).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or(IrError::InvalidValue {
                operation: "building an operation",
            })?;
        let value = self.graph.append(op, result, &operands).ok_or_else(|| {
            IrError::UnsupportedOperation {
                operation: format!("{op:?}"),
            }
        })?;
        let id = self.values.push(value);
        Ok(id)
    }
}

impl IrGraph {
    fn append(&mut self, operation: &Op, result: &TensorType, operands: &[Value]) -> Option<Value> {
        supported_dtype(result.dtype)?;
        macro_rules! unary {
            ($name:ident) => {{
                let [input] = operands else { return None };
                let ty = self.tensor_type_for(result);
                let op = construct_op!($name, &mut self.ctx, vec![ty], vec![*input]);
                self.push(op)
            }};
        }
        macro_rules! binary {
            ($name:ident) => {{
                let [lhs, rhs] = operands else { return None };
                let ty = self.tensor_type_for(result);
                let op = construct_op!($name, &mut self.ctx, vec![ty], vec![*lhs, *rhs]);
                self.push(op)
            }};
        }
        Some(match operation {
            Op::Parameter(number) => self.parameter_typed(*number, result),
            Op::StateInput {
                number,
                state_id,
                path,
            } => self.state_input(*number, *state_id, path, result),
            Op::StateRead { state_id } => {
                let [current] = operands else { return None };
                self.state_read(*current, *state_id)
            }
            Op::StateWrite { state_id } => {
                let [value] = operands else { return None };
                self.state_write(*value, *state_id)
            }
            Op::CustomCall {
                target,
                backend_config,
                has_side_effect,
                api_version,
                result_dtype: _,
            } => {
                let ty = self.tensor_type_for(result);
                let op = construct_op!(CustomCallOp, &mut self.ctx, vec![ty], operands.to_vec());
                op.set_attr_custom_call_target(&self.ctx, StringAttr::new(target.clone()));
                op.set_attr_custom_call_backend_config(
                    &self.ctx,
                    StringAttr::new(backend_config.clone()),
                );
                op.set_attr_custom_call_has_side_effect(
                    &self.ctx,
                    StringAttr::new(has_side_effect.to_string()),
                );
                op.set_attr_custom_call_api_version(
                    &self.ctx,
                    StringAttr::new(api_version.to_string()),
                );
                self.push(op)
            }
            Op::ConstantF32(value) => self.constant_f32(&result.dims, value.to_vec()),
            Op::ConstantI32(value) => self.constant_i32(&result.dims, value.to_vec()),
            Op::Iota { axis } => self.iota_typed(result, *axis),
            Op::GetDimensionSize { axis } => {
                let [input] = operands else { return None };
                let ty = self.tensor_type(&[], DType::I32);
                let op = construct_op!(GetDimensionSizeOp, &mut self.ctx, vec![ty], vec![*input]);
                op.set_attr_dimension_size_axis(&self.ctx, AxisAttr::new(*axis));
                self.push(op)
            }
            Op::IntegerBinary(operation) => {
                let [lhs, rhs] = operands else { return None };
                self.integer_binary_typed(*lhs, *rhs, result, *operation)
            }
            Op::Binary(Binary::Add) => {
                let [lhs, rhs] = operands else { return None };
                self.add_typed(*lhs, *rhs, result)
            }
            Op::Binary(Binary::Mul) => {
                let [lhs, rhs] = operands else { return None };
                self.multiply_typed(*lhs, *rhs, result)
            }
            Op::Binary(Binary::Sub) => binary!(SubtractOp),
            Op::Binary(Binary::Div) => binary!(DivideOp),
            Op::Binary(Binary::Maximum) => binary!(MaximumOp),
            Op::Binary(Binary::Minimum) => binary!(MinimumOp),
            Op::Matmul { batch_rank } => {
                let [lhs, rhs] = operands else { return None };
                self.matmul_typed(*lhs, *rhs, result, *batch_rank)
            }
            Op::Attention { scale } => {
                if !matches!(operands, [_, _, _] | [_, _, _, _]) {
                    return None;
                }
                let ty = self.tensor_type_for(result);
                let op = construct_op!(AttentionOp, &mut self.ctx, vec![ty], operands.to_vec());
                op.set_attr_attention_scale(&self.ctx, AttentionScaleAttr::new(*scale));
                self.push(op)
            }
            Op::Conv2d(options) => {
                let [input, kernel] = operands else {
                    return None;
                };
                self.conv2d_typed(*input, *kernel, result, *options)
            }
            Op::Conv2dOihw(options) => {
                let [input, kernel] = operands else {
                    return None;
                };
                self.conv2d_oihw_typed(*input, *kernel, result, *options)
            }
            Op::ConvTranspose2d(options) => {
                let [input, kernel] = operands else {
                    return None;
                };
                self.conv_transpose2d_typed(*input, *kernel, result, *options)
            }
            Op::MaxPool2d(options) => {
                let [input] = operands else { return None };
                self.pool2d_typed(*input, result, *options, true)
            }
            Op::SumPool2d(options) => {
                let [input] = operands else { return None };
                self.pool2d_typed(*input, result, *options, false)
            }
            Op::Unary(Unary::Floor) => unary!(FloorOp),
            Op::Unary(Unary::Ceil) => unary!(CeilOp),
            Op::Unary(Unary::Round) => unary!(RoundOp),
            Op::Unary(Unary::RoundTiesEven) => unary!(RoundTiesEvenOp),
            Op::Unary(Unary::Sin) => unary!(SinOp),
            Op::Unary(Unary::Cos) => unary!(CosOp),
            Op::Unary(Unary::Erf) => unary!(ErfOp),
            Op::Unary(Unary::Exp) => unary!(ExpOp),
            Op::Unary(Unary::Log) => unary!(LogOp),
            Op::Unary(Unary::Log1p) => unary!(Log1pOp),
            Op::Unary(Unary::Expm1) => unary!(Expm1Op),
            Op::Unary(Unary::Abs) => unary!(AbsOp),
            Op::Unary(Unary::Neg) => unary!(NegateOp),
            Op::Unary(Unary::Sqrt) => unary!(SqrtOp),
            Op::Unary(Unary::Rsqrt) => unary!(RsqrtOp),
            Op::Unary(Unary::Tanh) => unary!(TanhOp),
            Op::Relu => unary!(ReluOp),
            Op::Softplus => unary!(SoftplusOp),
            Op::Sigmoid => unary!(SigmoidOp),
            Op::StopGradient => unary!(StopGradientOp),
            Op::OptimizationBarrier => unary!(OptimizationBarrierOp),
            Op::WithGradient => {
                let [_, _] = operands else { return None };
                self.gradient_wrapper_typed(operands.to_vec(), result, false)
            }
            Op::WithElementwiseDerivative => {
                if operands.len() < 3 || operands.len().is_multiple_of(2) {
                    return None;
                }
                self.gradient_wrapper_typed(operands.to_vec(), result, true)
            }
            Op::Reshape => unary!(ReshapeOp),
            Op::IndexToFloat => unary!(IndexToFloatOp),
            Op::Bf16ToFloat => unary!(Bf16ToFloatOp),
            Op::Convert { .. } => unary!(ConvertOp),
            Op::Broadcast { axes } => {
                let [input] = operands else { return None };
                self.broadcast_typed(*input, result, axes.clone())
            }
            Op::Transpose { permutation } => {
                let [input] = operands else { return None };
                self.transpose_typed(*input, result, permutation.clone())
            }
            Op::Reverse { axes } => {
                let [input] = operands else { return None };
                self.reverse_typed(*input, result, axes.clone())
            }
            Op::Cumsum { axis } => {
                let [input] = operands else { return None };
                self.cumsum_typed(*input, result, *axis)
            }
            Op::Slice(spec) => {
                let [input] = operands else { return None };
                self.slice_typed(*input, result, spec.clone())
            }
            Op::SliceGradient(spec) => {
                let [input] = operands else { return None };
                self.slice_gradient_typed(*input, result, spec.clone())
            }
            Op::Pad(padding) => {
                let [input, fill] = operands else { return None };
                self.pad_typed(*input, *fill, result, padding.clone())
            }
            Op::DynamicSlice => self.dynamic_slice_typed(operands.to_vec(), result),
            Op::DynamicUpdateSlice => self.dynamic_update_slice_typed(operands.to_vec(), result),
            Op::Take { axis } => {
                let [input, indices] = operands else {
                    return None;
                };
                self.take_typed(*input, *indices, result, *axis)
            }
            Op::TakeAlongAxis { axis } => {
                let [input, indices] = operands else {
                    return None;
                };
                self.take_along_axis_typed(*input, *indices, result, *axis)
            }
            Op::GatherGradient { axis, batched } => {
                let [gradient, indices] = operands else {
                    return None;
                };
                self.gather_gradient_typed(*gradient, *indices, result, *axis, *batched)
            }
            Op::Concatenate { axis } => self.concatenate_typed(operands.to_vec(), result, *axis),
            Op::CompareMask(comparison) => {
                let [lhs, rhs] = operands else { return None };
                self.compare_mask_typed(*lhs, *rhs, result, *comparison)
            }
            Op::IndexLessEqualMask => {
                let [lhs, rhs] = operands else { return None };
                self.compare_mask_typed(*lhs, *rhs, result, Comparison::LessEqual)
            }
            Op::IsFiniteMask => {
                let [input] = operands else { return None };
                self.is_finite_mask_typed(*input, result)
            }
            Op::Select => {
                let [mask, on_true, on_false] = operands else {
                    return None;
                };
                self.select_typed(*mask, *on_true, *on_false, result)
            }
            Op::If { .. } | Op::MultiResult { .. } => return None,
            Op::Reduce {
                kind: Reduction::Sum,
                axes,
            } => {
                let [input] = operands else { return None };
                self.reduce_sum_typed(*input, result, axes.clone())
            }
            Op::Reduce {
                kind: Reduction::Maximum,
                axes,
            } => {
                let [input] = operands else { return None };
                self.reduce_maximum_typed(*input, result, axes.clone())
            }
            Op::ArgMax { axis } => {
                let [input] = operands else { return None };
                self.argmax_typed(*input, result, *axis)
            }
            Op::SortedIndices { axis, descending } => {
                let [input] = operands else { return None };
                self.sorted_indices_typed(*input, result, *axis, *descending)
            }
            Op::Conv2dKernelGradient(options) => {
                let [input, output_gradient] = operands else {
                    return None;
                };
                self.conv2d_kernel_gradient_typed(*input, *output_gradient, result, *options)
            }
            Op::Conv2dInputGradient(_) => {
                return None;
            }
        })
    }
}
