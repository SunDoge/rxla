use super::*;
use rxla_ir::IntegerBinary;

fn unbind_shape(shape: &[i64], axis: usize) -> Result<(i64, Vec<i64>)> {
    let &count = shape
        .get(axis)
        .ok_or_else(|| err("unbind axis out of range"))?;
    let mut dims = shape.to_vec();
    dims.remove(axis);
    Ok((count, dims))
}

fn index_shape(dims: &[i64]) -> TensorType {
    TensorType {
        dims: dims.to_vec(),
        dtype: DType::I32,
    }
}

fn slice_spec(
    shape: &[i64],
    starts: &[i64],
    limits: &[i64],
    strides: &[i64],
) -> Result<(Vec<i64>, Vec<SliceAxis>)> {
    let rank = shape.len();
    if starts.len() != rank || limits.len() != rank || strides.len() != rank {
        return Err(err("slice requires one start, limit and stride per axis"));
    }
    let mut dims = Vec::with_capacity(rank);
    let mut slices = Vec::with_capacity(rank);
    for axis in 0..rank {
        let (start, limit, stride) = (starts[axis], limits[axis], strides[axis]);
        if start < 0 || limit < start || limit > shape[axis] || stride <= 0 {
            return Err(err(format!(
                "invalid slice bounds or stride at axis {axis}"
            )));
        }
        let span = limit - start;
        // Avoid overflow from (span + stride - 1) / stride.
        dims.push(span / stride + i64::from(span % stride != 0));
        slices.push(SliceAxis {
            start,
            limit,
            stride,
        });
    }
    Ok((dims, slices))
}

fn narrow_spec(
    shape: &[i64],
    axis: usize,
    start: i64,
    length: i64,
) -> Result<(Vec<i64>, Vec<SliceAxis>)> {
    if axis >= shape.len() || length < 0 {
        return Err(err("invalid narrow axis or length"));
    }
    let end = start
        .checked_add(length)
        .ok_or_else(|| err("narrow bounds overflow"))?;
    let mut starts = vec![0; shape.len()];
    let mut limits = shape.to_vec();
    starts[axis] = start;
    limits[axis] = end;
    slice_spec(shape, &starts, &limits, &vec![1; shape.len()])
}

fn validate_take_along_axis(
    graph: &Graph,
    shape: &[i64],
    indices: &Tensor,
    axis: usize,
) -> Result<()> {
    if indices.dtype() != DType::I32 {
        return Err(err("gather indices must be I32"));
    }
    if !Arc::ptr_eq(&graph.0, &indices.graph().0) {
        return Err(err("cross-graph index"));
    }
    if axis >= shape.len() || shape[axis] == 0 {
        return Err(err("take_along_axis requires a nonempty valid axis"));
    }
    if shape.len() != indices.shape.len()
        || shape
            .iter()
            .zip(indices.shape.iter())
            .enumerate()
            .any(|(i, (a, b))| i != axis && a != b)
    {
        return Err(err(
            "take_along_axis requires equal rank and non-axis dimensions",
        ));
    }
    Ok(())
}

fn validate_dynamic_indices(
    graph: &Graph,
    shape: &[i64],
    starts: &[Tensor],
    sizes: &[i64],
) -> Result<()> {
    if starts.len() != shape.len() || sizes.len() != shape.len() {
        return Err(err("dynamic slicing requires one index and size per axis"));
    }
    if starts.iter().any(|i| i.dtype() != DType::I32) {
        return Err(err("dynamic slice starts must be I32"));
    }
    if starts.iter().any(|i| !i.shape.is_empty()) {
        return Err(err("dynamic slice starts must be scalar"));
    }
    if starts.iter().any(|i| !Arc::ptr_eq(&graph.0, &i.graph().0)) {
        return Err(err("cross-graph index"));
    }
    if sizes
        .iter()
        .zip(shape)
        .any(|(&size, &dim)| size < 0 || size > dim)
    {
        return Err(err("dynamic slice size must be within operand dimensions"));
    }
    Ok(())
}

impl Tensor {
    fn integer_binary(&self, rhs: &Self, op: IntegerBinary) -> Result<Self> {
        if self.dtype() != DType::I32 || rhs.dtype() != DType::I32 {
            return Err(err("integer bitwise operands must be I32"));
        }
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) || self.shape != rhs.shape {
            return Err(err("integer bitwise operands must match graph and shape"));
        }
        let id = self.graph().push_node(
            Op::IntegerBinary(op),
            vec![self.node_id(), rhs.node_id()],
            index_shape(self.shape()),
        )?;
        Ok(Tensor::symbolic(
            self.graph().clone(),
            id,
            self.shape(),
            DType::I32,
        ))
    }

    /// Elementwise AND on I32 bit patterns, without floating-point conversion.
    /// Operand graphs/shapes must match; broadcast explicitly. Nondifferentiable.
    pub fn bitwise_and(&self, rhs: &Self) -> Result<Self> {
        self.integer_binary(rhs, IntegerBinary::And)
    }
    /// Elementwise OR on same-shaped I32 bit patterns.
    pub fn bitwise_or(&self, rhs: &Self) -> Result<Self> {
        self.integer_binary(rhs, IntegerBinary::Or)
    }
    /// Elementwise XOR on same-shaped I32 bit patterns.
    pub fn bitwise_xor(&self, rhs: &Self) -> Result<Self> {
        self.integer_binary(rhs, IntegerBinary::Xor)
    }
    /// Complement every bit, preserving I32 shape and representation.
    pub fn bitwise_not(&self) -> Result<Self> {
        self.bitwise_xor(&self.graph().scalar_i32(-1)?.broadcast_to(self.shape())?)
    }
    fn shift(&self, bits: u32, op: IntegerBinary) -> Result<Self> {
        if bits >= 32 {
            return Err(err("I32 shift count must be less than 32"));
        }
        self.integer_binary(
            &self
                .graph()
                .scalar_i32(bits as i32)?
                .broadcast_to(self.shape())?,
            op,
        )
    }
    /// Left shift by a construction-time count in 0..32; high bits are discarded.
    /// Rejects out-of-range counts rather than silently reducing them modulo 32.
    pub fn shift_left(&self, bits: u32) -> Result<Self> {
        self.shift(bits, IntegerBinary::ShiftLeft)
    }
    /// Right shift with zero fill, including for negative I32 inputs.
    /// Count must be in 0..32. The result remains an I32 bit pattern.
    pub fn shift_right_logical(&self, bits: u32) -> Result<Self> {
        self.shift(bits, IntegerBinary::ShiftRightLogical)
    }
    /// Right shift with sign extension; count must be in 0..32.
    pub fn shift_right_arithmetic(&self, bits: u32) -> Result<Self> {
        self.shift(bits, IntegerBinary::ShiftRightArithmetic)
    }

    /// Explicit I32/BF16-to-F32 conversion (not a bitcast); F32 is identity.
    /// Finite BF16 values widen exactly. Integers through
    /// +/-2^24 are exactly representable; larger values may round. Keep counters
    /// and checkpoint state in I32, converting only for floating-point arithmetic.
    /// Integer construction paths are nondifferentiable, even if an index was
    /// selected using floating-point data such as argmax.
    pub fn to_f32(&self) -> Result<Tensor> {
        match self.dtype() {
            DType::F32 => Ok(self.clone()),
            DType::I32 => self
                .graph()
                .node(Op::IndexToFloat, vec![self.node_id()], self.shape()),
            DType::BF16 => self
                .graph()
                .node(Op::Bf16ToFloat, vec![self.node_id()], self.shape()),
            dtype => Err(err(format!("cannot convert dtype {dtype:?} to F32"))),
        }
    }
    fn arithmetic(&self, rhs: &Self, op: Binary) -> Result<Self> {
        if self.dtype() != DType::I32 || rhs.dtype() != DType::I32 {
            return Err(err("wrapping arithmetic operands must be I32"));
        }
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) {
            return Err(err("cross-graph index arithmetic"));
        }
        if self.shape != rhs.shape {
            return Err(err("index arithmetic shape mismatch; broadcast explicitly"));
        }
        let id = self.graph().push_node(
            Op::Binary(op),
            vec![self.node_id(), rhs.node_id()],
            index_shape(&self.shape),
        )?;
        Ok(Tensor::symbolic(
            self.graph().clone(),
            id,
            self.shape(),
            DType::I32,
        ))
    }
    /// Elementwise I32 addition modulo 2^32. Shapes must match; broadcast
    /// explicitly. Never converts through F32 or checks/saturates overflow.
    pub fn wrapping_add(&self, rhs: &Self) -> Result<Self> {
        self.arithmetic(rhs, Binary::Add)
    }
    /// Elementwise I32 subtraction modulo 2^32, with equal input shapes.
    pub fn wrapping_sub(&self, rhs: &Self) -> Result<Self> {
        self.arithmetic(rhs, Binary::Sub)
    }
    /// Elementwise I32 multiplication modulo 2^32, with equal input shapes.
    pub fn wrapping_mul(&self, rhs: &Self) -> Result<Self> {
        self.arithmetic(rhs, Binary::Mul)
    }
    /// Offset every index by a scalar, wrapping on I32 overflow.
    /// Slice/gather operations still apply their separate index-clamping rules.
    pub fn wrapping_add_scalar(&self, value: i32) -> Result<Self> {
        self.wrapping_add(&self.graph().scalar_i32(value)?.broadcast_to(self.shape())?)
    }
}

impl Graph {
    /// Construct I32 coordinates along one axis: each element contains its index
    /// on that axis, replicated over other dimensions. Uses native HLO iota,
    /// without constructing/uploading a host literal array. Shape/axis are static;
    /// dimensions must be nonnegative and axis length at most i32::MAX + 1.
    /// Empty tensors are allowed, but scalars have no valid coordinate axis.
    pub fn iota_i32(&self, dims: &[i64], axis: usize) -> Result<Tensor> {
        elements(dims)?;
        if axis >= dims.len() || dims[axis] > i32::MAX as i64 + 1 {
            return Err(err(
                "iota requires a valid axis with I32-representable coordinates",
            ));
        }
        let id = self.push_node(Op::Iota { axis }, vec![], index_shape(dims))?;
        Ok(Tensor::symbolic(self.clone(), id, dims, DType::I32))
    }

    /// Add a scalar I32 parameter in the same argument order as tensor inputs.
    pub fn input_i32_scalar(&self) -> Result<Tensor> {
        self.input_i32(&[])
    }
    pub fn input_i32(&self, dims: &[i64]) -> Result<Tensor> {
        self.input_dtype(dims, DType::I32)
    }
    pub fn scalar_i32(&self, value: i32) -> Result<Tensor> {
        self.constant_i32(&[], &[value])
    }
    pub fn constant_i32(&self, dims: &[i64], values: &[i32]) -> Result<Tensor> {
        if elements(dims)? != values.len() {
            return Err(err("index constant shape/data mismatch"));
        }
        let id = self.push_node(Op::ConstantI32(values.into()), vec![], index_shape(dims))?;
        Ok(Tensor::symbolic(self.clone(), id, dims, DType::I32))
    }
}

impl Tensor {
    /// Remove an axis and return its slices in ascending index order, the
    /// inverse shape operation of stack. An empty axis returns an empty Vec;
    /// scalars and absent axes are rejected. Uses differentiable static
    /// slice/reshape nodes, not host downloads or a promised zero-copy view.
    pub fn unbind(&self, axis: usize) -> Result<Vec<Self>> {
        let (count, dims) = unbind_shape(self.shape(), axis)?;
        (0..count)
            .map(|i| self.narrow(axis, i, 1)?.reshape(&dims))
            .collect()
    }

    /// Index of the first maximum along an axis, returned as an I32 tensor. Ties choose
    /// the lowest index; if a row contains NaNs, its first NaN wins. The reduced
    /// axis must be nonempty and its length fit in I32. Other empty axes are valid.
    /// Records reductions/comparisons, not a guaranteed fused argmax kernel.
    pub fn argmax(&self, axis: usize, keep_dims: bool) -> Result<Tensor> {
        if axis >= self.shape.len() || self.shape[axis] <= 0 || self.shape[axis] > i32::MAX as i64 {
            return Err(err("argmax requires a nonempty axis with I32-sized length"));
        }
        let mut dims = self.shape.to_vec();
        dims.remove(axis);
        let id = self.graph().push_node(
            Op::ArgMax { axis },
            vec![self.node_id()],
            index_shape(&dims),
        )?;
        let value = Tensor::symbolic(self.graph().clone(), id, &dims, DType::I32);
        if keep_dims {
            dims.insert(axis, 1);
            value.reshape(&dims)
        } else {
            Ok(value)
        }
    }
    /// Insert a size-one axis at `axis` (0 through rank, inclusive).
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.shape.len() {
            return Err(err("unsqueeze axis out of range"));
        }
        let mut dims = self.shape.to_vec();
        dims.insert(axis, 1);
        self.reshape(&dims)
    }
    /// Remove one explicitly selected size-one axis. Other singleton axes are
    /// preserved; removing a non-singleton or absent axis is an error.
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        if self.shape.get(axis) != Some(&1) {
            return Err(err("squeeze requires a valid size-one axis"));
        }
        let mut dims = self.shape.to_vec();
        dims.remove(axis);
        self.reshape(&dims)
    }
    /// Join equal-shaped tensors along a new axis (0 through input rank).
    /// Unlike concatenate, this increases rank by one. Inputs must be nonempty
    /// and belong to one graph; scalars and zero-sized tensors are supported.
    pub fn stack(tensors: &[Self], axis: usize) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| err("stack requires tensors"))?;
        if axis > first.shape.len() {
            return Err(err("stack axis out of range"));
        }
        for tensor in tensors {
            if tensor.shape != first.shape {
                return Err(err("stack requires identical input shapes"));
            }
            if !Arc::ptr_eq(&first.graph().0, &tensor.graph().0) {
                return Err(err("cross-graph stack operands"));
            }
        }
        let expanded = tensors
            .iter()
            .map(|t| t.unsqueeze(axis))
            .collect::<Result<Vec<_>>>()?;
        Self::concatenate(&expanded, axis)
    }
    /// Constant edge padding, one `[low, high]` pair per axis, preserving layout.
    /// Both widths must be nonnegative; use `slice`/`narrow` for cropping.
    /// Padding an empty dimension is allowed. NaN/infinite fill values are
    /// allowed and affect only padded elements. No interior padding is inserted.
    pub fn pad(&self, padding: &[[i64; 2]], value: f32) -> Result<Self> {
        let dims = self.padding_shape(padding)?;
        if dims == self.shape.as_ref() {
            return Ok(self.clone());
        }
        let fill = self.graph().constant(&[], &[value])?;
        self.graph().node(
            Op::Pad(padding.to_vec()),
            vec![self.node_id(), fill.node_id()],
            &dims,
        )
    }

    /// Edge padding using a same-graph rank-zero F32 tensor as fill. Both the
    /// input and the scalar are differentiable. No cropping/interior padding.
    /// Zero-width padding returns the input after validating the scalar.
    pub fn pad_with_scalar(&self, padding: &[[i64; 2]], fill: &Tensor) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &fill.graph().0) || !fill.shape().is_empty() {
            return Err(err("padding fill must be a same-graph scalar"));
        }
        let dims = self.padding_shape(padding)?;
        if dims == self.shape.as_ref() {
            return Ok(self.clone());
        }
        self.graph().node(
            Op::Pad(padding.to_vec()),
            vec![self.node_id(), fill.node_id()],
            &dims,
        )
    }

    fn padding_shape(&self, padding: &[[i64; 2]]) -> Result<Vec<i64>> {
        if padding.len() != self.shape.len() {
            return Err(err("pad requires one low/high pair per axis"));
        }
        let dims = self
            .shape
            .iter()
            .zip(padding)
            .map(|(&dim, &[low, high])| {
                if low < 0 || high < 0 {
                    return Err(err(
                        "pad widths must be nonnegative; use slice for cropping",
                    ));
                }
                dim.checked_add(low)
                    .and_then(|d| d.checked_add(high))
                    .ok_or_else(|| err("padded dimension overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        elements(&dims)?;
        Ok(dims)
    }

    /// Replicate the nearest edge value, with nonnegative `[low, high]` widths
    /// per axis. A padded axis must be nonempty; unpadded empty axes are allowed.
    /// Built from slices, broadcasts and concatenation, including corner values.
    /// Gradients accumulate all copies into their source edge elements.
    pub fn pad_replicate(&self, padding: &[[i64; 2]]) -> Result<Self> {
        self.padding_shape(padding)?;
        // Validate every axis before adding any graph operations.
        if self
            .shape
            .iter()
            .zip(padding)
            .any(|(&size, &[low, high])| size == 0 && (low != 0 || high != 0))
        {
            return Err(err("cannot replicate padding from an empty axis"));
        }
        let mut result = self.clone();
        for (axis, &[low, high]) in padding.iter().enumerate() {
            if low == 0 && high == 0 {
                continue;
            }
            let mut pieces = Vec::with_capacity(3);
            let mut dims = result.shape().to_vec();
            if low > 0 {
                dims[axis] = low;
                pieces.push(result.narrow(axis, 0, 1)?.broadcast_to(&dims)?);
            }
            pieces.push(result.clone());
            if high > 0 {
                dims[axis] = high;
                pieces.push(
                    result
                        .narrow(axis, result.shape()[axis] - 1, 1)?
                        .broadcast_to(&dims)?,
                );
            }
            result = Self::concatenate(&pieces, axis)?;
        }
        Ok(result)
    }

    /// Reflect across the edge without repeating the edge element. Every
    /// nonzero width must be smaller than that axis's original dimension.
    /// For `[a,b,c]`, widths `[1,2]` produce `[b,a,b,c,b,a]`.
    /// Unpadded empty/singleton axes are valid. Differentiable via slice/reverse.
    pub fn pad_reflect(&self, padding: &[[i64; 2]]) -> Result<Self> {
        self.padding_shape(padding)?;
        if self
            .shape
            .iter()
            .zip(padding)
            .any(|(&size, &[low, high])| (low > 0 && low >= size) || (high > 0 && high >= size))
        {
            return Err(err(
                "reflection widths must be smaller than the source dimension",
            ));
        }
        let mut result = self.clone();
        for (axis, &[low, high]) in padding.iter().enumerate() {
            if low == 0 && high == 0 {
                continue;
            }
            let mut pieces = Vec::with_capacity(3);
            if low > 0 {
                pieces.push(result.narrow(axis, 1, low)?.flip(&[axis])?);
            }
            pieces.push(result.clone());
            if high > 0 {
                pieces.push(
                    result
                        .narrow(axis, result.shape()[axis] - high - 1, high)?
                        .flip(&[axis])?,
                );
            }
            result = Self::concatenate(&pieces, axis)?;
        }
        Ok(result)
    }
    /// Select separately at each non-axis position. Input and indices must have
    /// equal rank and equal non-axis dimensions; broadcasting is explicit.
    /// The result has the indices' shape. Unlike `take`, indices are not shared
    /// across batches. Indices clamp to [0, axis_size-1]; negatives do not wrap.
    /// For logits [B,S,V], indices [B,S,1] select one value per token.
    pub fn take_along_axis(&self, indices: &Tensor, axis: usize) -> Result<Self> {
        validate_take_along_axis(self.graph(), self.shape(), indices, axis)?;
        self.graph().node(
            Op::TakeAlongAxis { axis },
            vec![self.node_id(), indices.node_id()],
            &indices.shape,
        )
    }
    /// Largest `k` values and their original I32 indices along `axis`, sorted
    /// descending. Equal values (including signed zeros) preserve input order;
    /// NaNs sort last and preserve their order. Requires 0 <= k <= axis size.
    /// Both outputs replace the axis size with k. Non-axis empty dimensions work.
    /// Values differentiate through the selected entries, not through the choice
    /// of indices; ties use the selected original entries, not averaged gradients.
    /// Lowers to one stable XLA sort plus slice/gather, not k unrolled reductions.
    /// Backend optimization determines whether the full sort can be avoided.
    /// k=0 returns empty values/indices without sorting, including an empty
    /// source axis. The value derivative is zero for every input entry.
    pub fn topk(&self, k: usize, axis: usize) -> Result<(Self, Tensor)> {
        if axis >= self.shape.len()
            || self.shape[axis] > i32::MAX as i64
            || k as u128 > self.shape[axis] as u128
        {
            return Err(err(
                "topk requires a valid I32-sized axis and k <= axis size",
            ));
        }
        let mut dims = self.shape.to_vec();
        dims[axis] = k as i64;
        if k == 0 {
            return Ok((
                self.narrow(axis, 0, 0)?,
                self.graph().iota_i32(&dims, axis)?,
            ));
        }
        let id = self.graph().push_node(
            Op::SortedIndices {
                axis,
                descending: true,
            },
            vec![self.node_id()],
            index_shape(self.shape()),
        )?;
        let indices = Tensor::symbolic(self.graph().clone(), id, self.shape(), DType::I32)
            .narrow(axis, 0, k as i64)?;
        Ok((self.take_along_axis(&indices, axis)?, indices))
    }

    /// Stable ordering indices, with the same shape as this tensor. Set
    /// descending=false for ascending values, true for descending values.
    /// NaNs are last in BOTH directions. Equal values (including signed zeros)
    /// and NaNs retain original index order. Empty axes are supported; axis
    /// must be valid and I32-sized. Integer results are nondifferentiable.
    /// Use take_along_axis to reorder F32 or I32 payloads by these indices.
    pub fn argsort(&self, axis: usize, descending: bool) -> Result<Tensor> {
        if axis >= self.shape.len() || self.shape[axis] > i32::MAX as i64 {
            return Err(err("argsort requires a valid I32-sized axis"));
        }
        if self.shape[axis] == 0 {
            return self.graph().iota_i32(self.shape(), axis);
        }
        let id = self.graph().push_node(
            Op::SortedIndices { axis, descending },
            vec![self.node_id()],
            index_shape(self.shape()),
        )?;
        Ok(Tensor::symbolic(
            self.graph().clone(),
            id,
            self.shape(),
            DType::I32,
        ))
    }
    /// Gather along one axis, replacing that axis with the index tensor's shape.
    /// Indices clamp to [0, axis_size-1]; negatives do not wrap. For embedding,
    /// table.take(token_ids, 0) maps [B,S] IDs into [B,S,hidden] values.
    pub fn take(&self, indices: &Tensor, axis: usize) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &indices.graph().0) {
            return Err(err("cross-graph index"));
        }
        if axis >= self.shape.len() || self.shape[axis] == 0 {
            return Err(err("take requires a nonempty valid axis"));
        }
        let mut dims = self.shape[..axis].to_vec();
        dims.extend_from_slice(&indices.shape);
        dims.extend_from_slice(&self.shape[axis + 1..]);
        self.graph().node(
            Op::Take { axis },
            vec![self.node_id(), indices.node_id()],
            &dims,
        )
    }
    /// Extract a fixed-size slice at runtime indices. XLA clamps each index to
    /// [0, dimension - size]; negative indices do not count from the end.
    pub fn dynamic_slice(&self, starts: &[Tensor], sizes: &[i64]) -> Result<Self> {
        validate_dynamic_indices(self.graph(), self.shape(), starts, sizes)?;
        let mut operands = vec![self.node_id()];
        operands.extend(starts.iter().map(|i| i.node_id()));
        self.graph().node(Op::DynamicSlice, operands, sizes)
    }

    /// Return a tensor with the specified region replaced. Indices are clamped
    /// as in `dynamic_slice`. This does not mutate or donate the input buffer.
    pub fn dynamic_update_slice(&self, update: &Self, starts: &[Tensor]) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &update.graph().0) {
            return Err(err("cross-graph update"));
        }
        validate_dynamic_indices(self.graph(), self.shape(), starts, update.shape())?;
        let mut operands = vec![self.node_id(), update.node_id()];
        operands.extend(starts.iter().map(|i| i.node_id()));
        self.graph()
            .node(Op::DynamicUpdateSlice, operands, &self.shape)
    }

    /// Rebind this handle to a tensor with one contiguous region replaced.
    ///
    /// This emits `stablehlo.dynamic_update_slice`; clones of the old handle
    /// retain the old SSA value. Backend buffer reuse remains an XLA
    /// alias/donation decision rather than observable Rust mutation.
    pub fn slice_copy_(&mut self, update: &Self, starts: &[Tensor]) -> Result<&mut Self> {
        *self = self.dynamic_update_slice(update, starts)?;
        Ok(self)
    }

    /// Add indexed updates along one axis using StableHLO scatter-add.
    /// Indices clamp exactly like [`Self::take`], and repeated indices sum.
    pub fn index_add(&self, axis: usize, indices: &Tensor, updates: &Self) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &indices.graph().0)
            || !Arc::ptr_eq(&self.graph().0, &updates.graph().0)
        {
            return Err(err("cross-graph indexed update"));
        }
        if self.dtype() != DType::F32
            || updates.dtype() != DType::F32
            || indices.dtype() != DType::I32
            || axis >= self.shape().len()
            || self.shape()[axis] == 0
        {
            return Err(err(
                "index_add requires F32 tensors, I32 indices and a nonempty valid axis",
            ));
        }
        let mut expected = self.shape()[..axis].to_vec();
        expected.extend_from_slice(indices.shape());
        expected.extend_from_slice(&self.shape()[axis + 1..]);
        if updates.shape() != expected {
            return Err(err(
                "index_add update shape does not match indexed result shape",
            ));
        }
        let scattered = self.graph().node(
            Op::GatherGradient {
                axis,
                batched: false,
            },
            vec![updates.node_id(), indices.node_id()],
            self.shape(),
        )?;
        self.add(&scattered)
    }

    /// Rebind this handle to the result of [`Self::index_add`].
    pub fn index_add_(
        &mut self,
        axis: usize,
        indices: &Tensor,
        updates: &Self,
    ) -> Result<&mut Self> {
        *self = self.index_add(axis, indices, updates)?;
        Ok(self)
    }

    /// Static, half-open slicing on every axis. Strides must be positive;
    /// negative indexing and Python-style bound clipping are not performed.
    pub fn slice(&self, starts: &[i64], limits: &[i64], strides: &[i64]) -> Result<Self> {
        let (dims, slices) = slice_spec(self.shape(), starts, limits, strides)?;
        self.graph()
            .node(Op::Slice(slices), vec![self.node_id()], &dims)
    }

    /// Select a contiguous region along one axis; the selected axis is retained.
    pub fn narrow(&self, axis: usize, start: i64, length: i64) -> Result<Self> {
        let (dims, slices) = narrow_spec(self.shape(), axis, start, length)?;
        self.graph()
            .node(Op::Slice(slices), vec![self.node_id()], &dims)
    }

    /// Circularly shift one axis; positive shifts move toward higher indices.
    /// Shift is a graph-construction constant, reduced modulo the axis length.
    /// Empty axes and multiples of the length are no-ops. Invalid axes fail
    /// even for a zero shift. Compose calls to shift multiple axes.
    pub fn roll(&self, shift: i64, axis: usize) -> Result<Self> {
        let &size = self
            .shape()
            .get(axis)
            .ok_or_else(|| err("roll axis out of range"))?;
        if size == 0 {
            return Ok(self.clone());
        }
        let shift = shift.rem_euclid(size);
        if shift == 0 {
            return Ok(self.clone());
        }
        Self::concatenate(
            &[
                self.narrow(axis, size - shift, shift)?,
                self.narrow(axis, 0, size - shift)?,
            ],
            axis,
        )
    }

    /// Repeat each element along an axis consecutively, preserving other axes.
    /// For example `[a, b]` repeated twice becomes `[a, a, b, b]`, not tiled
    /// `[a, b, a, b]`. The count is a nonnegative graph-construction constant;
    /// zero produces an empty axis. The axis must exist even for zero/one count.
    /// Uses reshape/broadcast, not a host loop or a guaranteed materialized copy.
    /// Gradients sum contributions from each repeated element.
    pub fn repeat_interleave(&self, repeats: i64, axis: usize) -> Result<Self> {
        let &size = self
            .shape
            .get(axis)
            .ok_or_else(|| err("repeat_interleave axis out of range"))?;
        if repeats < 0 {
            return Err(err("repeat_interleave count must be nonnegative"));
        }
        let mut output = self.shape.to_vec();
        output[axis] = size
            .checked_mul(repeats)
            .ok_or_else(|| err("repeat_interleave dimension overflow"))?;
        elements(&output)?;
        if repeats == 1 {
            return Ok(self.clone());
        }
        let mut expanded = self.shape.to_vec();
        expanded.insert(axis + 1, repeats);
        self.unsqueeze(axis + 1)?
            .broadcast_to(&expanded)?
            .reshape(&output)
    }

    /// Split an axis into explicitly sized pieces. Sizes must cover it exactly.
    pub fn split(&self, axis: usize, sizes: &[i64]) -> Result<Vec<Self>> {
        if axis >= self.shape.len() || sizes.is_empty() {
            return Err(err("split requires a valid axis and nonempty sizes"));
        }
        let total = sizes.iter().try_fold(0i64, |sum, &size| {
            if size < 0 {
                None
            } else {
                sum.checked_add(size)
            }
        });
        if total != Some(self.shape[axis]) {
            return Err(err("split sizes must sum to the axis length"));
        }
        let mut start = 0;
        sizes
            .iter()
            .map(|&size| {
                let result = self.narrow(axis, start, size);
                start += size;
                result
            })
            .collect()
    }

    /// Concatenate tensors along an existing axis. Other dimensions must match.
    pub fn concatenate(tensors: &[Self], axis: usize) -> Result<Self> {
        let first = tensors
            .first()
            .ok_or_else(|| err("concatenate requires tensors"))?;
        if axis >= first.shape.len() {
            return Err(err("concatenate axis out of range"));
        }
        let mut dims = first.shape.to_vec();
        dims[axis] = 0;
        for tensor in tensors {
            if !Arc::ptr_eq(&first.graph().0, &tensor.graph().0) {
                return Err(err("cross-graph operands"));
            }
            if tensor.shape.len() != dims.len()
                || tensor
                    .shape
                    .iter()
                    .zip(&dims)
                    .enumerate()
                    .any(|(i, (a, b))| i != axis && a != b)
            {
                return Err(err("concatenate non-axis dimensions must match"));
            }
            dims[axis] = dims[axis]
                .checked_add(tensor.shape[axis])
                .ok_or_else(|| err("concatenate dimension overflow"))?;
        }
        if tensors.len() == 1 {
            return Ok(first.clone());
        }
        first.graph().node(
            Op::Concatenate { axis },
            tensors.iter().map(|t| t.node_id()).collect(),
            &dims,
        )
    }
}

#[cfg(test)]
mod inplace_tests {
    use super::*;

    #[test]
    fn inplace_tensor_handles_preserve_alias_ssa_values() {
        let mut base = Tensor::from_slice([4, 2], DType::F32, [0.0; 8]).unwrap();
        let alias = base.clone();
        let update = Tensor::from_slice([1, 2], DType::F32, [7.0, 8.0]).unwrap();
        let row = Tensor::from_slice([], DType::I32, [2]).unwrap();
        let column = Tensor::from_slice([], DType::I32, [0]).unwrap();
        base.slice_copy_(&update, &[row, column]).unwrap();

        assert_ne!(base.node_id(), alias.node_id());

        let mut scatter_base = Tensor::from_slice([4, 2], DType::F32, [0.0; 8]).unwrap();
        let scatter_alias = scatter_base.clone();
        let indices = Tensor::from_slice([3], DType::I32, [1, 1, 3]).unwrap();
        let updates = Tensor::from_slice([3, 2], DType::F32, [1.0; 6]).unwrap();
        scatter_base.index_add_(0, &indices, &updates).unwrap();
        assert_ne!(scatter_base.node_id(), scatter_alias.node_id());
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn inplace_tensor_updates_execute_through_the_tensor_api() {
        let mut runtime =
            unsafe { Runtime::load(std::env::var("PJRT_PLUGIN_PATH").expect("PJRT_PLUGIN_PATH")) }
                .unwrap();
        let mut sliced = Tensor::from_slice([4, 2], DType::F32, [0.0; 8]).unwrap();
        let update = Tensor::from_slice([1, 2], DType::F32, [7.0, 8.0]).unwrap();
        let row = Tensor::from_slice([], DType::I32, [2]).unwrap();
        let column = Tensor::from_slice([], DType::I32, [0]).unwrap();
        sliced.slice_copy_(&update, &[row, column]).unwrap();

        let mut scattered = Tensor::from_slice([4, 2], DType::F32, [0.0; 8]).unwrap();
        let indices = Tensor::from_slice([3], DType::I32, [1, 1, 3]).unwrap();
        let updates = Tensor::from_slice([3, 2], DType::F32, [1.0; 6]).unwrap();
        scattered.index_add_(0, &indices, &updates).unwrap();

        let outputs = runtime.eval_many(&[sliced, scattered]).unwrap();
        assert_eq!(
            outputs[0].to_vec::<f32>().unwrap(),
            [0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0]
        );
        assert_eq!(
            outputs[1].to_vec::<f32>().unwrap(),
            [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0]
        );
    }
}
