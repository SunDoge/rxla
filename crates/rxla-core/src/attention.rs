use super::*;

impl Tensor {
    /// Additive causal mask from explicit query `[Q]` and key `[K]` positions.
    /// Entry `(i,j)` is zero iff `key_positions[j] <= self[i]`, else -infinity.
    /// Comparison is signed I32, with no subtraction or float conversion; even
    /// extreme I32 positions are exact. Negative/duplicate/unsorted positions
    /// are compared as supplied, not treated as padding or validated cache slots.
    ///
    /// Runtime position buffers can vary without recompilation at fixed shapes.
    /// The mask stops gradients through positions, but attention Q/K/V gradients
    /// remain available. All-masked rows retain ordinary undefined softmax
    /// behavior; this does not track cache validity or suppress padding itself.
    pub fn causal_attention_mask(&self, key_positions: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::I32 || key_positions.dtype() != DType::I32 {
            return Err(err("causal positions must be I32 tensors"));
        }
        if self.shape().len() != 1 || key_positions.shape().len() != 1 {
            return Err(err("causal positions must be rank-one I32 tensors"));
        }
        if !Arc::ptr_eq(&self.graph().0, &key_positions.graph().0) {
            return Err(err("cross-graph causal positions"));
        }
        let shape = [self.shape()[0], key_positions.shape()[0]];
        let query = self.reshape(&[shape[0], 1])?.broadcast_to(&shape)?;
        key_positions
            .reshape(&[1, shape[1]])?
            .broadcast_to(&shape)?
            .le_mask(&query)?
            .log()?
            .detach()
    }
}

impl Graph {
    /// Additive F32 causal mask [queries, keys]: key j is visible to row i iff
    /// j <= query_offset + i. Visible entries are zero, others negative infinity.
    /// Offset zero is ordinary prefill; a positive offset supports a later chunk
    /// or one-token decode against a cache indexed from position zero. Broadcast
    /// over batch/head dimensions explicitly or pass to an attention method.
    ///
    /// Counts and offset are static, nonnegative and must fit I32 coordinates.
    /// Empty masks are allowed. This does not update a KV cache, track runtime
    /// positions, infer padding or guarantee a fused attention kernel.
    pub fn causal_attention_mask(
        &self,
        queries: i64,
        keys: i64,
        query_offset: i64,
    ) -> Result<Tensor> {
        let max = i64::from(i32::MAX);
        if queries < 0
            || keys < 0
            || keys > max + 1
            || query_offset < 0
            || query_offset > max
            || queries > max + 1 - query_offset
        {
            return Err(err(
                "causal mask requires nonnegative I32-representable positions",
            ));
        }
        let query = self
            .iota_i32(&[queries], 0)?
            .wrapping_add_scalar(query_offset as i32)?;
        query.causal_attention_mask(&self.iota_i32(&[keys], 0)?)
    }
}

impl Tensor {
    /// Grouped-query attention without explicit K/V head repetition.
    /// Q is `[..., query_heads, queries, depth]`; K/V are
    /// `[..., kv_heads, keys, depth/value_depth]`. Ranks and batch prefixes must
    /// match exactly; positive query_heads must be divisible by positive kv_heads.
    /// Consecutive query heads share one KV head. Output restores query layout
    /// with value_depth in the last dimension.
    ///
    /// Mask broadcasts to `[..., query_heads, queries, keys]` BEFORE grouping,
    /// so per-query and per-head biases retain their intended ordering. Query
    /// grouping is reshape-only; mask broadcasting/reshape may still require
    /// backend materialization. Scale/masking semantics are identical to SDPA.
    /// No implicit causal mask, dropout or fused-kernel guarantee is provided.
    pub fn grouped_query_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: Option<f32>,
    ) -> Result<Tensor> {
        let q = self.shape();
        let k = key.shape();
        let v = value.shape();
        if q.len() < 3 || q.len() != k.len() || q.len() != v.len() {
            return Err(err("GQA requires matching ranks >= 3"));
        }
        let head = q.len() - 3;
        if q[..head] != k[..head]
            || q[..head] != v[..head]
            || q[head] <= 0
            || k[head] <= 0
            || v[head] != k[head]
            || q[head] % k[head] != 0
        {
            return Err(err(
                "GQA requires equal batch prefixes and divisible positive head counts",
            ));
        }
        let rows = q[head + 1]
            .checked_mul(q[head] / k[head])
            .ok_or_else(|| err("GQA query row overflow"))?;
        let mut grouped = q.to_vec();
        grouped[head] = k[head];
        grouped[head + 1] = rows;
        let grouped_mask = if let Some(mask) = mask {
            if !Arc::ptr_eq(&self.graph().0, &mask.graph().0) {
                return Err(err("cross-graph GQA mask"));
            }
            let mut full_scores = q.to_vec();
            full_scores[head + 2] = k[head + 1];
            let mut grouped_scores = grouped.clone();
            grouped_scores[head + 2] = k[head + 1];
            Some(mask.broadcast_to(&full_scores)?.reshape(&grouped_scores)?)
        } else {
            None
        };
        let result = self.reshape(&grouped)?.scaled_dot_product_attention(
            key,
            value,
            grouped_mask.as_ref(),
            scale,
        )?;
        let mut output = q.to_vec();
        output[head + 2] = v[head + 2];
        result.reshape(&output)
    }

    /// Scaled dot-product attention: `softmax(Q K^T * scale + mask) V`.
    /// Shapes: Q `[..., queries, depth]`, K `[..., keys, depth]`, V
    /// `[..., keys, value_depth]`. Leading dimensions use matmul broadcasting;
    /// this does not repeat/reshape grouped-query heads automatically.
    ///
    /// Scale defaults to `1/sqrt(depth)`; an explicit scale must be finite.
    /// Mask is additive F32 and broadcasts to the score shape: 0 keeps an entry,
    /// -infinity masks it. There is no implicit causal mask, boolean conversion,
    /// dropout, or KV update. Every row needs a finite maximum for defined
    /// softmax behavior; all-masked rows are not silently replaced with zeros.
    ///
    /// This records ordinary matmul/softmax ops for backend optimization, NOT a
    /// guaranteed fused or memory-efficient FlashAttention kernel.
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: Option<f32>,
    ) -> Result<Tensor> {
        let q = self.shape();
        let k = key.shape();
        let v = value.shape();
        if q.len() < 2 || k.len() < 2 || v.len() < 2 {
            return Err(err("attention requires rank >= 2 Q/K/V"));
        }
        if self.dtype() != DType::F32
            || key.dtype() != DType::F32
            || value.dtype() != DType::F32
            || mask.is_some_and(|mask| mask.dtype() != DType::F32)
        {
            return Err(err("attention requires F32 Q/K/V and mask tensors"));
        }
        let query_type = self.ty();
        let key_type = key.ty();
        let value_type = value.ty();
        let depth = q[q.len() - 1];
        if depth <= 0
            || k[k.len() - 1] != depth
            || key_type.bound(k.len() - 1) != query_type.bound(q.len() - 1)
            || k[k.len() - 2] == 0
            || k[k.len() - 2] != v[v.len() - 2]
            || key_type.bound(k.len() - 2) != value_type.bound(v.len() - 2)
        {
            return Err(err(
                "attention requires matching positive Q/K depth and K/V length",
            ));
        }
        for tensor in [Some(key), Some(value), mask].into_iter().flatten() {
            if !Arc::ptr_eq(&self.graph().0, &tensor.graph().0) {
                return Err(err("cross-graph attention operand"));
            }
        }
        let scale = scale.unwrap_or_else(|| 1. / (depth as f32).sqrt());
        if !scale.is_finite() {
            return Err(err("attention scale must be finite"));
        }
        let batch_rank = (q.len() - 2).max(k.len() - 2).max(v.len() - 2);
        let mut batch_dims = vec![1; batch_rank];
        let mut batch_bounds = vec![-1; batch_rank];
        for operand in [&query_type, &key_type, &value_type] {
            let operand_batch_rank = operand.dims.len() - 2;
            let offset = batch_rank - operand_batch_rank;
            for axis in 0..operand_batch_rank {
                let target_axis = offset + axis;
                let dimension = operand.dims[axis];
                let bound = operand.bound(axis).unwrap_or(-1);
                if batch_dims[target_axis] == 1 {
                    batch_dims[target_axis] = dimension;
                    batch_bounds[target_axis] = bound;
                } else if dimension != 1
                    && (dimension != batch_dims[target_axis] || bound != batch_bounds[target_axis])
                {
                    return Err(err("attention batch dimensions cannot broadcast"));
                }
            }
        }
        let broadcast_operand = |tensor: &Tensor, rows_axis: usize, columns_axis: usize| {
            let source = tensor.ty();
            let mut dims = batch_dims.clone();
            dims.extend([source.dims[rows_axis], source.dims[columns_axis]]);
            let mut bounds = batch_bounds.clone();
            bounds.extend([
                source.bound(rows_axis).unwrap_or(-1),
                source.bound(columns_axis).unwrap_or(-1),
            ]);
            tensor.broadcast_to_type(&inferred_tensor_type(dims, DType::F32, bounds))
        };
        let query = broadcast_operand(self, q.len() - 2, q.len() - 1)?;
        let key = broadcast_operand(key, k.len() - 2, k.len() - 1)?;
        let value = broadcast_operand(value, v.len() - 2, v.len() - 1)?;
        let mut score_dims = batch_dims.clone();
        score_dims.extend([q[q.len() - 2], k[k.len() - 2]]);
        let mut score_bounds = batch_bounds.clone();
        score_bounds.extend([
            query_type.bound(q.len() - 2).unwrap_or(-1),
            key_type.bound(k.len() - 2).unwrap_or(-1),
        ]);
        let score_type = inferred_tensor_type(score_dims, DType::F32, score_bounds);
        let mut operands = vec![query.node_id(), key.node_id(), value.node_id()];
        if let Some(mask) = mask {
            operands.push(mask.broadcast_to_type(&score_type)?.node_id());
        }
        let mut output_dims = batch_dims;
        output_dims.extend([q[q.len() - 2], v[v.len() - 1]]);
        let mut output_bounds = batch_bounds;
        output_bounds.extend([
            query_type.bound(q.len() - 2).unwrap_or(-1),
            value_type.bound(v.len() - 1).unwrap_or(-1),
        ]);
        self.graph().node_typed(
            rxla_ir::Op::Attention { scale },
            operands,
            inferred_tensor_type(output_dims, DType::F32, output_bounds),
        )
    }
}
