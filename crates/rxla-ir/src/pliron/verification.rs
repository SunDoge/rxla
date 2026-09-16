//! Semantic verification for framework operations.

use super::*;

fn value_type(value: Value, ctx: &Context) -> pliron::result::Result<TensorType> {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let Some(ranked) = ty.downcast_ref::<RankedTensorType>() else {
        return pliron::verify_err_noloc!("rxla operations require ranked tensor values");
    };
    ranked.verify(ctx)?;
    Ok(TensorType {
        dims: ranked.shape.values(),
        dtype: ranked.element.value(),
    })
}

fn compatible_dimension(lhs: i64, rhs: i64) -> bool {
    lhs == -1 || rhs == -1 || lhs == rhs
}

fn static_elements(dims: &[i64]) -> Option<u128> {
    dims.iter().try_fold(1u128, |elements, &dimension| {
        (dimension >= 0)
            .then_some(dimension as u128)
            .and_then(|dimension| elements.checked_mul(dimension))
    })
}

impl Verify for ParameterOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(number) = self.get_attr_number(ctx) else {
            return pliron::verify_err_noloc!("rxla.parameter requires an ABI number");
        };
        if number.as_str().parse::<usize>().is_err() {
            return pliron::verify_err_noloc!(
                "rxla.parameter ABI number must be a nonnegative integer"
            );
        }
        Ok(())
    }
}

impl Verify for StateInputOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(number) = self.get_attr_state_number(ctx) else {
            return pliron::verify_err_noloc!("rxla.state_input requires an ABI number");
        };
        let Some(state_id) = self.get_attr_input_state_id(ctx) else {
            return pliron::verify_err_noloc!("rxla.state_input requires a state id");
        };
        let Some(path) = self.get_attr_state_path(ctx) else {
            return pliron::verify_err_noloc!("rxla.state_input requires a state path");
        };
        if number.as_str().parse::<usize>().is_err()
            || state_id.as_str().parse::<usize>().is_err()
            || path.as_str().is_empty()
        {
            return pliron::verify_err_noloc!(
                "rxla.state_input requires numeric ABI/state ids and a nonempty path"
            );
        }
        Ok(())
    }
}

impl Verify for StateReadOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let _ = value_type(self.get_operation().deref(ctx).get_operand(0), ctx)?;
        let Some(state_id) = self.get_attr_read_state_id(ctx) else {
            return pliron::verify_err_noloc!("rxla.state_read requires a state id");
        };
        if state_id.as_str().parse::<usize>().is_err() {
            return pliron::verify_err_noloc!("rxla.state_read state id must be numeric");
        }
        Ok(())
    }
}

impl Verify for StateWriteOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let _ = value_type(self.get_operation().deref(ctx).get_operand(0), ctx)?;
        let Some(state_id) = self.get_attr_write_state_id(ctx) else {
            return pliron::verify_err_noloc!("rxla.state_write requires a state id");
        };
        if state_id.as_str().parse::<usize>().is_err() {
            return pliron::verify_err_noloc!("rxla.state_write state id must be numeric");
        }
        Ok(())
    }
}

impl Verify for IfOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let result_count = operation.get_num_results();
        let predicate = value_type(operation.get_operand(0), ctx)?;
        if predicate.dtype != DType::I32 || !predicate.dims.is_empty() {
            return pliron::verify_err_noloc!("rxla.if predicate must be a scalar I32 value");
        }
        if operation.num_regions() != 2 {
            return pliron::verify_err_noloc!("rxla.if requires exactly two regions");
        }
        for region_index in 0..2 {
            let region = operation.get_region(region_index);
            let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();
            if blocks.len() != 1 || !blocks[0].deref(ctx).arguments().next().is_none() {
                return pliron::verify_err_noloc!(
                    "rxla.if regions must contain exactly one block without arguments"
                );
            }
            let Some(yield_op) = blocks[0].deref(ctx).get_tail() else {
                return pliron::verify_err_noloc!("rxla.if regions must end with rxla.yield");
            };
            if !Operation::is_op::<YieldOp>(yield_op, ctx) {
                return pliron::verify_err_noloc!("rxla.if regions must end with rxla.yield");
            }
            let yielded = yield_op.deref(ctx);
            if yielded.get_num_operands() != result_count {
                return pliron::verify_err_noloc!(
                    "rxla.if yield arity must equal conditional result arity"
                );
            }
            for index in 0..result_count {
                let yielded_value = yielded.get_operand(index);
                if value_type(yielded_value, ctx)? != value_type(operation.get_result(index), ctx)?
                {
                    return pliron::verify_err_noloc!(
                        "rxla.if yield types must equal conditional result types"
                    );
                }
            }
        }
        Ok(())
    }
}

impl Verify for ConstantOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(value) = self.get_attr_value(ctx) else {
            return pliron::verify_err_noloc!("rxla.constant requires a value payload");
        };
        let result = value_type(self.get_operation().deref(ctx).get_result(0), ctx)?;
        let Some(elements) = static_elements(&result.dims) else {
            return pliron::verify_err_noloc!(
                "rxla.constant requires a static, non-overflowing shape"
            );
        };
        let element_bytes = match result.dtype {
            DType::U8 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            _ => return pliron::verify_err_noloc!("rxla.constant has an unsupported dtype"),
        };
        if elements.checked_mul(element_bytes) != Some(value.as_ref().len() as u128) {
            return pliron::verify_err_noloc!(
                "rxla.constant payload size does not match its result type"
            );
        }
        Ok(())
    }
}

impl Verify for IotaOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(axis) = self.get_attr_iota_axis(ctx) else {
            return pliron::verify_err_noloc!("rxla.iota requires an axis attribute");
        };
        if result.dtype != DType::I32 {
            return pliron::verify_err_noloc!("rxla.iota result must have I32 elements");
        }
        if axis.value() >= result.dims.len() {
            return pliron::verify_err_noloc!("rxla.iota axis is outside the result rank");
        }
        Ok(())
    }
}

impl Verify for AttentionOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        if !matches!(operation.get_num_operands(), 3 | 4) {
            return pliron::verify_err_noloc!("rxla.attention requires Q, K, V and optional mask");
        }
        let query = value_type(operation.get_operand(0), ctx)?;
        let key = value_type(operation.get_operand(1), ctx)?;
        let value = value_type(operation.get_operand(2), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        if query.dtype != DType::F32
            || key.dtype != DType::F32
            || value.dtype != DType::F32
            || result.dtype != DType::F32
            || query.dims.len() < 2
            || query.dims.len() != key.dims.len()
            || query.dims.len() != value.dims.len()
        {
            return pliron::verify_err_noloc!(
                "rxla.attention requires equal-rank F32 Q, K and V tensors"
            );
        }
        let rank = query.dims.len();
        if query.dims[..rank - 2] != key.dims[..rank - 2]
            || query.dims[..rank - 2] != value.dims[..rank - 2]
            || !compatible_dimension(query.dims[rank - 1], key.dims[rank - 1])
            || !compatible_dimension(key.dims[rank - 2], value.dims[rank - 2])
        {
            return pliron::verify_err_noloc!("rxla.attention has incompatible Q/K/V shapes");
        }
        let mut expected_result = query.dims.clone();
        expected_result[rank - 1] = value.dims[rank - 1];
        if result.dims != expected_result {
            return pliron::verify_err_noloc!("rxla.attention result shape is inconsistent");
        }
        if operation.get_num_operands() == 4 {
            let mask = value_type(operation.get_operand(3), ctx)?;
            let mut expected_mask = query.dims.clone();
            expected_mask[rank - 1] = key.dims[rank - 2];
            if mask.dtype != DType::F32 || mask.dims != expected_mask {
                return pliron::verify_err_noloc!(
                    "rxla.attention mask must match the broadcast score shape"
                );
            }
        }
        let Some(scale) = self.get_attr_attention_scale(ctx) else {
            return pliron::verify_err_noloc!("rxla.attention requires a scale attribute");
        };
        if !scale.value().is_finite() {
            return pliron::verify_err_noloc!("rxla.attention scale must be finite");
        }
        Ok(())
    }
}

impl Verify for ReshapeOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        if input.dtype != result.dtype {
            return pliron::verify_err_noloc!("rxla.reshape must preserve element type");
        }
        if let (Some(input), Some(result)) =
            (static_elements(&input.dims), static_elements(&result.dims))
            && input != result
        {
            return pliron::verify_err_noloc!("rxla.reshape must preserve element count");
        }
        Ok(())
    }
}

impl Verify for BroadcastOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(axes) = self.get_attr_broadcast_axes(ctx) else {
            return pliron::verify_err_noloc!("rxla.broadcast requires mapped axes");
        };
        let axes = axes.values();
        if input.dtype != result.dtype {
            return pliron::verify_err_noloc!("rxla.broadcast must preserve element type");
        }
        if axes.len() != input.dims.len()
            || axes.windows(2).any(|pair| pair[0] >= pair[1])
            || axes.iter().any(|&axis| axis >= result.dims.len())
        {
            return pliron::verify_err_noloc!(
                "rxla.broadcast axes must map every input axis in increasing order"
            );
        }
        for (input_axis, &result_axis) in axes.iter().enumerate() {
            let source = input.dims[input_axis];
            let target = result.dims[result_axis];
            if source != 1 && !compatible_dimension(source, target) {
                return pliron::verify_err_noloc!(
                    "rxla.broadcast mapped dimensions must match or broadcast from one"
                );
            }
        }
        Ok(())
    }
}

impl Verify for TransposeOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(permutation) = self.get_attr_permutation(ctx) else {
            return pliron::verify_err_noloc!("rxla.transpose requires a permutation");
        };
        let permutation = permutation.values();
        let mut sorted = permutation.clone();
        sorted.sort_unstable();
        if input.dtype != result.dtype {
            return pliron::verify_err_noloc!("rxla.transpose must preserve element type");
        }
        if permutation.len() != input.dims.len()
            || result.dims.len() != input.dims.len()
            || sorted != (0..input.dims.len()).collect::<Vec<_>>()
        {
            return pliron::verify_err_noloc!(
                "rxla.transpose permutation must contain every input axis exactly once"
            );
        }
        if permutation
            .iter()
            .enumerate()
            .any(|(result_axis, &input_axis)| {
                !compatible_dimension(result.dims[result_axis], input.dims[input_axis])
            })
        {
            return pliron::verify_err_noloc!(
                "rxla.transpose result shape does not match its permutation"
            );
        }
        Ok(())
    }
}

fn verify_scalar_indices(
    operation: &Operation,
    ctx: &Context,
    first: usize,
    rank: usize,
) -> pliron::result::Result<()> {
    if operation.get_num_operands() != first + rank {
        return pliron::verify_err_noloc!(
            "dynamic slice operations require one scalar index per tensor axis"
        );
    }
    for index in first..operation.get_num_operands() {
        let index = value_type(operation.get_operand(index), ctx)?;
        if index.dtype != DType::I32 || !index.dims.is_empty() {
            return pliron::verify_err_noloc!("dynamic slice indices must be scalar I32 values");
        }
    }
    Ok(())
}

impl Verify for DynamicSliceOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        if operation.get_num_operands() == 0 {
            return pliron::verify_err_noloc!("rxla.dynamic_slice requires an input tensor");
        }
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        verify_scalar_indices(&operation, ctx, 1, input.dims.len())?;
        if input.dtype != result.dtype || input.dims.len() != result.dims.len() {
            return pliron::verify_err_noloc!(
                "rxla.dynamic_slice must preserve element type and rank"
            );
        }
        if input
            .dims
            .iter()
            .zip(&result.dims)
            .any(|(&input, &result)| input >= 0 && result >= 0 && result > input)
        {
            return pliron::verify_err_noloc!(
                "rxla.dynamic_slice result cannot exceed its input shape"
            );
        }
        Ok(())
    }
}

impl Verify for DynamicUpdateSliceOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        if operation.get_num_operands() < 2 {
            return pliron::verify_err_noloc!(
                "rxla.dynamic_update_slice requires base and update tensors"
            );
        }
        let base = value_type(operation.get_operand(0), ctx)?;
        let update = value_type(operation.get_operand(1), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        verify_scalar_indices(&operation, ctx, 2, base.dims.len())?;
        if base.dtype != update.dtype
            || base.dtype != result.dtype
            || base.dims.len() != update.dims.len()
            || base.dims.len() != result.dims.len()
            || base
                .dims
                .iter()
                .zip(&result.dims)
                .any(|(&base, &result)| !compatible_dimension(base, result))
        {
            return pliron::verify_err_noloc!(
                "rxla.dynamic_update_slice result must match its base tensor"
            );
        }
        if base
            .dims
            .iter()
            .zip(&update.dims)
            .any(|(&base, &update)| base >= 0 && update >= 0 && update > base)
        {
            return pliron::verify_err_noloc!(
                "rxla.dynamic_update_slice update cannot exceed the base shape"
            );
        }
        Ok(())
    }
}

impl Verify for TakeOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let indices = value_type(operation.get_operand(1), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(axis) = self.get_attr_take_axis(ctx) else {
            return pliron::verify_err_noloc!("rxla.take requires an axis attribute");
        };
        let axis = axis.value();
        if axis >= input.dims.len() || input.dims[axis] == 0 {
            return pliron::verify_err_noloc!("rxla.take axis must select a nonempty dimension");
        }
        if indices.dtype != DType::I32 || result.dtype != input.dtype {
            return pliron::verify_err_noloc!(
                "rxla.take requires I32 indices and preserves the input element type"
            );
        }
        let expected = input.dims[..axis]
            .iter()
            .chain(&indices.dims)
            .chain(&input.dims[axis + 1..])
            .copied()
            .collect::<Vec<_>>();
        if expected.len() != result.dims.len()
            || expected
                .iter()
                .zip(&result.dims)
                .any(|(&expected, &result)| !compatible_dimension(expected, result))
        {
            return pliron::verify_err_noloc!("rxla.take result shape is inconsistent");
        }
        Ok(())
    }
}

impl Verify for TakeAlongAxisOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let indices = value_type(operation.get_operand(1), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(axis) = self.get_attr_take_along_axis(ctx) else {
            return pliron::verify_err_noloc!("rxla.take_along_axis requires an axis attribute");
        };
        let axis = axis.value();
        if axis >= input.dims.len() || input.dims[axis] == 0 {
            return pliron::verify_err_noloc!(
                "rxla.take_along_axis must select a nonempty dimension"
            );
        }
        if indices.dtype != DType::I32
            || result.dtype != input.dtype
            || indices.dims.len() != input.dims.len()
            || result.dims.len() != indices.dims.len()
        {
            return pliron::verify_err_noloc!(
                "rxla.take_along_axis requires equal ranks, I32 indices and matching result type"
            );
        }
        for dimension in 0..input.dims.len() {
            if !compatible_dimension(result.dims[dimension], indices.dims[dimension])
                || (dimension != axis
                    && !compatible_dimension(input.dims[dimension], indices.dims[dimension]))
            {
                return pliron::verify_err_noloc!(
                    "rxla.take_along_axis result and non-axis dimensions must match indices"
                );
            }
        }
        Ok(())
    }
}

impl Verify for GatherGradientOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let gradient = value_type(operation.get_operand(0), ctx)?;
        let indices = value_type(operation.get_operand(1), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(config) = self.get_attr_gather_gradient(ctx) else {
            return pliron::verify_err_noloc!("rxla.gather_gradient requires its configuration");
        };
        let (axis, batched) = config.values();
        if axis >= result.dims.len() || result.dims[axis] == 0 {
            return pliron::verify_err_noloc!(
                "rxla.gather_gradient axis must select a nonempty result dimension"
            );
        }
        if gradient.dtype != DType::F32 || result.dtype != DType::F32 || indices.dtype != DType::I32
        {
            return pliron::verify_err_noloc!(
                "rxla.gather_gradient requires F32 gradients/results and I32 indices"
            );
        }
        let expected = if batched {
            if indices.dims.len() != result.dims.len()
                || (0..result.dims.len()).any(|dimension| {
                    dimension != axis
                        && !compatible_dimension(result.dims[dimension], indices.dims[dimension])
                })
            {
                return pliron::verify_err_noloc!(
                    "batched gather gradients require equal rank and non-axis dimensions"
                );
            }
            indices.dims.clone()
        } else {
            result.dims[..axis]
                .iter()
                .chain(&indices.dims)
                .chain(&result.dims[axis + 1..])
                .copied()
                .collect()
        };
        if expected.len() != gradient.dims.len()
            || expected
                .iter()
                .zip(&gradient.dims)
                .any(|(&expected, &actual)| !compatible_dimension(expected, actual))
        {
            return pliron::verify_err_noloc!("rxla.gather_gradient input shape is inconsistent");
        }
        Ok(())
    }
}

impl Verify for CumsumOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(axis) = self.get_attr_axis(ctx) else {
            return pliron::verify_err_noloc!("rxla.cumsum requires an axis attribute");
        };
        if result.dtype != DType::F32 {
            return pliron::verify_err_noloc!("rxla.cumsum currently requires F32 values");
        }
        if axis.value() >= result.dims.len() || result.dims[axis.value()] <= 1 {
            return pliron::verify_err_noloc!(
                "rxla.cumsum axis must reference a statically nontrivial dimension"
            );
        }
        Ok(())
    }
}

impl Verify for SliceGradientOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(spec) = self.get_attr_gradient_spec(ctx) else {
            return pliron::verify_err_noloc!("rxla.slice_gradient requires a slice specification");
        };
        let spec = spec.values();
        if input.dtype != DType::F32 || result.dtype != DType::F32 {
            return pliron::verify_err_noloc!("rxla.slice_gradient requires F32 tensors");
        }
        if spec.len() != input.dims.len() || result.dims.len() != input.dims.len() {
            return pliron::verify_err_noloc!(
                "rxla.slice_gradient specification must match input and result rank"
            );
        }
        for (dimension, axis) in spec.iter().enumerate() {
            if axis.start < 0
                || axis.limit < axis.start
                || axis.stride <= 0
                || (result.dims[dimension] >= 0 && axis.limit > result.dims[dimension])
            {
                return pliron::verify_err_noloc!(
                    "rxla.slice_gradient contains an invalid slice range"
                );
            }
            let selected = if axis.limit == axis.start {
                0
            } else {
                (axis.limit - axis.start - 1) / axis.stride + 1
            };
            if input.dims[dimension] >= 0 && input.dims[dimension] != selected {
                return pliron::verify_err_noloc!(
                    "rxla.slice_gradient input shape does not match its slice range"
                );
            }
        }
        Ok(())
    }
}

impl Verify for ArgMaxOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(axis) = self.get_attr_argmax_axis(ctx) else {
            return pliron::verify_err_noloc!("rxla.argmax requires an axis attribute");
        };
        if input.dtype != DType::F32 || result.dtype != DType::I32 {
            return pliron::verify_err_noloc!("rxla.argmax requires F32 input and I32 output");
        }
        if axis.value() >= input.dims.len() {
            return pliron::verify_err_noloc!("rxla.argmax axis is out of range");
        }
        let expected = input
            .dims
            .iter()
            .enumerate()
            .filter_map(|(index, &dimension)| (index != axis.value()).then_some(dimension))
            .collect::<Vec<_>>();
        if result.dims != expected {
            return pliron::verify_err_noloc!("rxla.argmax result shape must remove its axis");
        }
        Ok(())
    }
}

impl Verify for SortedIndicesOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(sort) = self.get_attr_sort(ctx) else {
            return pliron::verify_err_noloc!("rxla.sorted_indices requires sort attributes");
        };
        let (axis, _) = sort.values();
        if input.dtype != DType::F32 || result.dtype != DType::I32 {
            return pliron::verify_err_noloc!(
                "rxla.sorted_indices requires F32 input and I32 output"
            );
        }
        if axis >= input.dims.len() || result.dims != input.dims {
            return pliron::verify_err_noloc!(
                "rxla.sorted_indices requires a valid axis and matching result shape"
            );
        }
        Ok(())
    }
}

impl Verify for ConvTranspose2dOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let operation = self.get_operation().deref(ctx);
        let input = value_type(operation.get_operand(0), ctx)?;
        let kernel = value_type(operation.get_operand(1), ctx)?;
        let result = value_type(operation.get_result(0), ctx)?;
        let Some(attribute) = self.get_attr_transpose_options(ctx) else {
            return pliron::verify_err_noloc!(
                "rxla.conv_transpose2d requires convolution attributes"
            );
        };
        let options = attribute.options();
        if input.dtype != DType::F32
            || kernel.dtype != DType::F32
            || result.dtype != DType::F32
            || input.dims.len() != 4
            || kernel.dims.len() != 4
            || result.dims.len() != 4
        {
            return pliron::verify_err_noloc!("rxla.conv_transpose2d requires rank-4 F32 tensors");
        }
        if input.dims[0] != result.dims[0]
            || input.dims[3] != kernel.dims[3]
            || kernel.dims[2] != result.dims[3]
        {
            return pliron::verify_err_noloc!(
                "rxla.conv_transpose2d batch or channel dimensions are inconsistent"
            );
        }
        for axis in 0..2 {
            if options.strides[axis] <= 0
                || options.dilation[axis] <= 0
                || options.padding[axis].iter().any(|&padding| padding < 0)
                || options.output_padding[axis] < 0
                || options.output_padding[axis] >= options.strides[axis]
            {
                return pliron::verify_err_noloc!(
                    "rxla.conv_transpose2d has invalid window attributes"
                );
            }
            if input.dims[axis + 1] >= 0 && kernel.dims[axis] >= 0 && result.dims[axis + 1] >= 0 {
                let expected = (input.dims[axis + 1] - 1) * options.strides[axis]
                    + (kernel.dims[axis] - 1) * options.dilation[axis]
                    + 1
                    - options.padding[axis][0]
                    - options.padding[axis][1]
                    + options.output_padding[axis];
                if expected != result.dims[axis + 1] {
                    return pliron::verify_err_noloc!(
                        "rxla.conv_transpose2d result shape is inconsistent"
                    );
                }
            }
        }
        Ok(())
    }
}

impl Verify for WithGradientOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        verify_gradient_wrapper(self.get_operation(), ctx, false)
    }
}

impl Verify for WithElementwiseDerivativeOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        verify_gradient_wrapper(self.get_operation(), ctx, true)
    }
}

fn verify_gradient_wrapper(
    operation: pliron::context::Ptr<Operation>,
    ctx: &Context,
    elementwise: bool,
) -> pliron::result::Result<()> {
    let operation = operation.deref(ctx);
    let operands = operation.operands().collect::<Vec<_>>();
    if operands.is_empty()
        || (elementwise && (operands.len() < 3 || operands.len().is_multiple_of(2)))
    {
        return pliron::verify_err_noloc!("rxla gradient wrapper has invalid operands");
    }
    let result = value_type(operation.get_result(0), ctx)?;
    for operand in operands {
        if value_type(operand, ctx)? != result {
            return pliron::verify_err_noloc!(
                "rxla gradient wrapper operands must match its result type"
            );
        }
    }
    Ok(())
}
