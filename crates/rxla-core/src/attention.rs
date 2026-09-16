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
        let depth = q[q.len() - 1];
        if depth <= 0
            || k[k.len() - 1] != depth
            || k[k.len() - 2] <= 0
            || k[k.len() - 2] != v[v.len() - 2]
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
        let mut batch = vec![1; batch_rank];
        for shape in [q, k, v] {
            for (axis, &dimension) in shape[..shape.len() - 2].iter().enumerate() {
                let target = &mut batch[batch_rank - (shape.len() - 2) + axis];
                if *target == 1 {
                    *target = dimension;
                } else if dimension != 1 && dimension != *target {
                    return Err(err("attention batch dimensions cannot broadcast"));
                }
            }
        }
        let broadcast_operand = |tensor: &Tensor, rows: i64, columns: i64| {
            let mut shape = batch.clone();
            shape.extend_from_slice(&[rows, columns]);
            tensor.broadcast_to(&shape)
        };
        let query = broadcast_operand(self, q[q.len() - 2], depth)?;
        let key = broadcast_operand(key, k[k.len() - 2], depth)?;
        let value = broadcast_operand(value, v[v.len() - 2], v[v.len() - 1])?;
        let mut score_shape = batch.clone();
        score_shape.extend_from_slice(&[q[q.len() - 2], k[k.len() - 2]]);
        let mut operands = vec![query.node_id(), key.node_id(), value.node_id()];
        if let Some(mask) = mask {
            operands.push(mask.broadcast_to(&score_shape)?.node_id());
        }
        let mut output = batch;
        output.extend_from_slice(&[q[q.len() - 2], v[v.len() - 1]]);
        self.graph()
            .node(rxla_ir::Op::Attention { scale }, operands, &output)
    }
}
