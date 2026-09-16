use super::{
    CategoricalSample, ThreefryBlocks, categorical_from_bits, normal_f32_from_bits,
    threefry2x32_blocks, top_p_categorical_from_bits, topk_categorical_from_bits,
    topk_sample_shape, uniform_f32_from_bits,
};
use crate::{Buffer, Client, Result, StateGraph, StateSlot, Tensor, err};
use std::sync::Arc;

/// Inverted-dropout output and its explicit F32 0/1 keep mask. Reuse keep_mask
/// with Tensor::dropout_with_mask when rebuilding a recomputation; drawing again
/// would reserve different random blocks. Neither field commits resident state.
pub struct DropoutSample {
    pub output: Tensor,
    pub keep_mask: Tensor,
}

/// Four resident scalar I32 slots, ordered key0, key1, counter-low, counter-high.
/// Clones share graph-local identities, not device buffers or independent streams.
#[derive(Clone)]
pub struct ThreefryState {
    slots: [StateSlot; 4],
}

impl crate::state_tree::StateTree for ThreefryState {
    /// Stable local names for exact key/counter words. Initial seed policy and
    /// checkpoint compatibility metadata remain explicit caller choices.
    fn visit_states(&self, visitor: &mut dyn FnMut(&str, &StateSlot)) {
        for (name, slot) in ["key0", "key1", "counter_low", "counter_high"]
            .into_iter()
            .zip(&self.slots)
        {
            visitor(name, slot);
        }
    }
}

impl ThreefryState {
    pub fn new(graph: &mut StateGraph) -> Result<Self> {
        Ok(Self {
            slots: [
                graph.state_i32(&[])?,
                graph.state_i32(&[])?,
                graph.state_i32(&[])?,
                graph.state_i32(&[])?,
            ],
        })
    }

    pub fn slots(&self) -> &[StateSlot; 4] {
        &self.slots
    }

    /// Explicitly replace the raw scalar key words and reset the counter to zero
    /// when condition is nonzero, using StateGraph's select semantics (NaN also
    /// accepts). All four slots update together. Shape/owner errors record no
    /// state writes. Existing sequences become stale after this symbolic write.
    ///
    /// Reusing a key intentionally repeats its stream; no entropy or uniqueness
    /// is guaranteed. When using keys from a parent sequence, pass the parent's
    /// RETURNED commit_if acceptance, so rejected/wrapped reservations cannot
    /// reset child state. This is a graph operation, not a host session reset.
    pub fn reset_key_if(
        &self,
        graph: &mut StateGraph,
        key: [&Tensor; 2],
        condition: &Tensor,
    ) -> Result<()> {
        let current = graph.read(&self.slots[0])?;
        for word in key {
            if !word.shape().is_empty() || !Arc::ptr_eq(&current.graph().0, &word.graph().0) {
                return Err(err("Threefry reset keys must be same-graph scalar words"));
            }
        }
        let zero = graph.scalar_i32(0)?;
        graph.write_many_if(
            condition,
            &[
                (&self.slots[0], key[0].clone()),
                (&self.slots[1], key[1].clone()),
                (&self.slots[2], zero.clone()),
                (&self.slots[3], zero),
            ],
        )
    }

    /// Explicit raw key words and initial counter; no seed hashing or entropy.
    /// Reusing a key and counter in another session intentionally repeats draws.
    pub fn initial_state(
        &self,
        client: &Client,
        key: [u32; 2],
        counter: u64,
    ) -> Result<Vec<(StateSlot, Buffer)>> {
        self.slots
            .iter()
            .zip([key[0], key[1], counter as u32, (counter >> 32) as u32])
            .map(|(slot, value)| Ok((slot.clone(), client.buffer(&[], &[value as i32])?)))
            .collect()
    }

    /// Begin a construction-time sequence from current symbolic state. Draws
    /// chain counters locally; dropping the sequence records no state writes.
    /// This is not a host RNG and these Rust methods do not run per device step.
    pub fn begin(&self, graph: &StateGraph) -> Result<ThreefrySequence> {
        let original = [
            graph.read(&self.slots[0])?,
            graph.read(&self.slots[1])?,
            graph.read(&self.slots[2])?,
            graph.read(&self.slots[3])?,
        ];
        Ok(ThreefrySequence {
            state: self.clone(),
            next: [original[2].clone(), original[3].clone()],
            available: graph.constant(&[], &[1.])?,
            original,
        })
    }
}

/// A non-Clone proposed sequence. Multiple draws reserve disjoint consecutive
/// blocks unless the 64-bit period is exhausted; any wrap rejects the entire
/// sequence at commit, even if subsequent draws no longer wrap.
pub struct ThreefrySequence {
    state: ThreefryState,
    original: [Tensor; 4],
    next: [Tensor; 2],
    available: Tensor,
}

impl ThreefrySequence {
    /// Nucleus sampling using one block per original logit, assigned in sorted
    /// rank order. Uses word0; filtered candidates still consume blocks.
    /// See top_p_categorical_from_bits for cutoff/validity rules. Failed
    /// construction preserves the proposed cursor. Reduce the returned valid
    /// mask and explicitly commit_if; use its acceptance for other state writes.
    pub fn top_p_categorical(
        &mut self,
        logits: &Tensor,
        p: f32,
        axis: usize,
    ) -> Result<CategoricalSample> {
        if !Arc::ptr_eq(&self.original[0].graph().0, &logits.graph().0) {
            return Err(err("top-p categorical input belongs to another RNG graph"));
        }
        let (draw, available) = self.propose(logits.shape())?;
        let sample = top_p_categorical_from_bits(logits, &draw.bits[0], p, axis)?;
        self.next = draw.next_counter;
        self.available = available;
        Ok(sample)
    }

    /// Top-k categorical draw returning original category IDs. Reserves one
    /// block per candidate (including k=1), not per original logit; uses word0.
    /// See `topk_categorical_from_bits` for validity and cutoff tie semantics.
    /// Failed construction leaves the proposed cursor unchanged. Reduce valid
    /// and pass it to commit_if, then guard other updates with its returned mask.
    pub fn topk_categorical(
        &mut self,
        logits: &Tensor,
        k: usize,
        axis: usize,
    ) -> Result<CategoricalSample> {
        if !Arc::ptr_eq(&self.original[0].graph().0, &logits.graph().0) {
            return Err(err("top-k categorical input belongs to another RNG graph"));
        }
        let shape = topk_sample_shape(logits, k, axis)?;
        let (draw, available) = self.propose(&shape)?;
        let sample = topk_categorical_from_bits(logits, &draw.bits[0], k, axis)?;
        self.next = draw.next_counter;
        self.available = available;
        Ok(sample)
    }

    /// Categorical sampling with one reserved Threefry block per logit, using
    /// word0. See `categorical_from_bits` for distribution and validity rules.
    /// Failed construction does not advance the proposed cursor. Invalid values
    /// at execution time still reserve blocks: reduce the returned `valid` mask
    /// and pass it to `commit_if` to reject advancement for invalid distributions.
    /// Use commit_if's returned acceptance for other state updates as well.
    pub fn categorical(&mut self, logits: &Tensor, axis: usize) -> Result<CategoricalSample> {
        if !Arc::ptr_eq(&self.original[0].graph().0, &logits.graph().0) {
            return Err(err("categorical input belongs to another RNG graph"));
        }
        let (draw, available) = self.propose(logits.shape())?;
        let sample = categorical_from_bits(logits, &draw.bits[0], axis)?;
        self.next = draw.next_counter;
        self.available = available;
        Ok(sample)
    }

    /// Training-only inverted dropout, using this sequence's Bernoulli policy.
    /// keep_probability is finite in (0, 1]; input must belong to this graph.
    /// For p<1, reserves one block per element and uses word0's high 24 bits.
    /// For p=1, returns the input and an all-one mask without consuming blocks.
    /// No evaluation mode is inferred: use the input directly for inference.
    ///
    /// Returned output differentiates through the input, not the random mask;
    /// autodiff reuses this draw rather than advancing the sequence again.
    /// Dropped nonfinite inputs are selected to zero before scaling. Because
    /// Bernoulli probabilities use a 24-bit grid, expected scaling may differ
    /// from exactly one for probabilities not representable on that grid.
    /// Errors do not advance the proposed cursor. Commit the sequence separately.
    pub fn dropout(&mut self, input: &Tensor, keep_probability: f32) -> Result<DropoutSample> {
        if !keep_probability.is_finite() || keep_probability <= 0. || keep_probability > 1. {
            return Err(err("dropout keep probability must be finite and in (0, 1]"));
        }
        if !Arc::ptr_eq(&self.original[0].graph().0, &input.graph().0) {
            return Err(err("dropout input belongs to another RNG graph"));
        }
        let graph = &self.original[0].graph();
        if keep_probability == 1. {
            return Ok(DropoutSample {
                output: input.clone(),
                keep_mask: graph.constant(&[], &[1.])?.broadcast_to(input.shape())?,
            });
        }
        let (draw, available) = self.propose(input.shape())?;
        let keep_mask = uniform_f32_from_bits(&draw.bits[0])?.lt_mask(
            &graph
                .constant(&[], &[keep_probability])?
                .broadcast_to(input.shape())?,
        )?;
        let output = input.dropout_with_mask(&keep_mask, keep_probability)?;
        self.next = draw.next_counter;
        self.available = available;
        Ok(DropoutSample { output, keep_mask })
    }

    /// Two word tensors, consuming one block per element (scalar: one, empty:
    /// zero). Failed construction leaves the proposed cursor unchanged.
    /// Values remain available for recomputation without making another draw.
    pub fn blocks(&mut self, shape: &[i64]) -> Result<[Tensor; 2]> {
        let (draw, available) = self.propose(shape)?;
        self.next = draw.next_counter;
        self.available = available;
        Ok(draw.bits)
    }

    /// Reserve `count` consecutive blocks as child key pairs, returned as two
    /// I32 tensors of shape [count] (word0, word1). This is the explicit policy
    /// `blocks([count])`, not JAX split compatibility or a new RNG algorithm.
    /// Zero count consumes nothing; count must fit I32. Errors preserve cursor.
    /// Commit this sequence separately and use its returned acceptance to guard
    /// child resets. Extract pair i with narrow/reshape; child counters start at
    /// zero when explicitly passed to ThreefryState::reset_key_if.
    ///
    /// No automatic child state, thread/replica assignment or domain separation
    /// from ordinary block draws exists. Reusing parent key/counter reproduces
    /// keys; no cross-tree uniqueness, statistical independence or cryptographic
    /// guarantee is made. Output keys from rejected reservations must not be used.
    pub fn split_keys(&mut self, count: usize) -> Result<[Tensor; 2]> {
        let count =
            i32::try_from(count).map_err(|_| err("Threefry split count exceeds I32 range"))?;
        self.blocks(&[i64::from(count)])
    }

    /// Uniform F32 values on the high-24-bit grid in [0, 1). Consumes one
    /// Threefry block per element, uses word0, and discards word1. Empty shapes
    /// consume none. Failed construction leaves the proposed cursor unchanged.
    pub fn uniform_f32(&mut self, shape: &[i64]) -> Result<Tensor> {
        self.sample(shape, None)
    }

    /// Approximate standard-normal noise; see `normal_f32_from_bits` for the
    /// finite-grid and rounding contract. Uses both words of one Threefry block
    /// per element, including scalar shapes; empty shapes consume none.
    /// Failed construction preserves the proposed cursor. Commit separately.
    pub fn normal_f32(&mut self, shape: &[i64]) -> Result<Tensor> {
        let (draw, available) = self.propose(shape)?;
        let output = normal_f32_from_bits([&draw.bits[0], &draw.bits[1]])?;
        self.next = draw.next_counter;
        self.available = available;
        Ok(output)
    }

    /// F32 0/1 mask with construction-time probability in [0, 1]. For interior
    /// probabilities, compares uniform_f32 < probability: effective probability
    /// is ceil(probability * 2^24) / 2^24, not the host Initializer's 64-bit rule.
    /// Each element consumes one block. Endpoints produce constants WITHOUT
    /// advancing the counter; empty shapes consume none. No inverted-dropout
    /// scaling. Invalid probabilities/shapes leave the proposed cursor unchanged.
    pub fn bernoulli(&mut self, shape: &[i64], probability: f32) -> Result<Tensor> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(err(
                "device Bernoulli probability must be finite and in [0, 1]",
            ));
        }
        if probability == 0. || probability == 1. {
            return self.original[0]
                .graph()
                .constant(&[], &[if probability == 0. { 0. } else { 1. }])?
                .broadcast_to(shape);
        }
        self.sample(shape, Some(probability))
    }

    fn sample(&mut self, shape: &[i64], probability: Option<f32>) -> Result<Tensor> {
        let (draw, available) = self.propose(shape)?;
        let uniform = uniform_f32_from_bits(&draw.bits[0])?;
        let output = match probability {
            Some(p) => uniform.lt_mask(
                &self.original[0]
                    .graph()
                    .constant(&[], &[p])?
                    .broadcast_to(shape)?,
            )?,
            None => uniform,
        };
        self.next = draw.next_counter;
        self.available = available;
        Ok(output)
    }

    fn propose(&self, shape: &[i64]) -> Result<(ThreefryBlocks, Tensor)> {
        let draw = threefry2x32_blocks(
            [&self.original[0], &self.original[1]],
            [&self.next[0], &self.next[1]],
            shape,
        )?;
        let available = self
            .available
            .mul(&draw.counter_wrapped.neg()?.add_scalar(1.)?)?;
        Ok((draw, available))
    }

    /// Record BOTH counter writes and return scalar F32 0/1 acceptance. A zero
    /// condition or ANY counter wrap rejects all advancement. Other condition
    /// values (including NaN) request acceptance: pass an explicit predicate.
    /// Use the RETURNED predicate for optimizer/statistic updates too.
    ///
    /// Consumes the proposal. Rejects stale proposals if any key/counter symbolic
    /// version changed since begin, and rejects another graph. On error no state
    /// versions change. This is not a transaction over other graph-building APIs.
    /// Rejected draws still compute; full-period exhaustion needs caller action.
    pub fn commit_if(self, graph: &mut StateGraph, condition: &Tensor) -> Result<Tensor> {
        for (slot, original) in self.state.slots.iter().zip(&self.original) {
            if graph.read(slot)?.node_id() != original.node_id() {
                return Err(err("stale Threefry sequence: state changed since begin"));
            }
        }
        let accepted = condition.select(&self.available, &graph.constant(&[], &[0.])?)?;
        graph.write_many_if(
            &accepted,
            &[
                (&self.state.slots[2], self.next[0].clone()),
                (&self.state.slots[3], self.next[1].clone()),
            ],
        )?;
        Ok(accepted)
    }
}
