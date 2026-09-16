//! Explicit resident BatchNorm running statistics, not trainable parameters.
use super::*;

/// Two graph-local running-stat slots. Clones share identities; independent
/// sessions still own independent buffers. No affine parameters or step counter
/// are registered, and optimizers do not discover these slots as parameters.
#[derive(Clone)]
pub struct BatchNormState {
    mean: StateSlot,
    variance: StateSlot,
    channels: usize,
}

impl crate::state_tree::StateTree for BatchNormState {
    /// Running statistics only; affine parameters and configuration are separate.
    fn visit_states(&self, visitor: &mut dyn FnMut(&str, &StateSlot)) {
        visitor("mean", &self.mean);
        visitor("variance", &self.variance);
    }
}

/// Non-Clone proposed EMA update. No statistic writes occur until commit.
/// Holding/dropping this object does not affect a live session or its buffers.
pub struct BatchNormUpdate {
    state: BatchNormState,
    original: [Tensor; 2],
    proposed: [Tensor; 2],
}
impl BatchNormState {
    pub fn new(graph: &mut StateGraph, channels: i64) -> Result<Self> {
        if channels <= 0 {
            return Err(err("BatchNorm state requires positive channels"));
        }
        let size =
            usize::try_from(channels).map_err(|_| err("BatchNorm channel count overflow"))?;
        Ok(Self {
            mean: graph.state(&[channels])?,
            variance: graph.state(&[channels])?,
            channels: size,
        })
    }
    /// Explicit initial buffers: running mean zero, running variance one.
    /// Allocates/uploads host arrays; does not initialize other program state.
    pub fn initial_state(&self, client: &Client) -> Result<Vec<(StateSlot, Buffer)>> {
        let shape = [self.channels as i64];
        Ok(vec![
            (
                self.mean.clone(),
                client.buffer(&shape, &vec![0.; self.channels])?,
            ),
            (
                self.variance.clone(),
                client.buffer(&shape, &vec![1.; self.channels])?,
            ),
        ])
    }
    pub fn mean_slot(&self) -> &StateSlot {
        &self.mean
    }
    pub fn variance_slot(&self) -> &StateSlot {
        &self.variance
    }
    /// Current symbolic versions; call after recording updates to use the newly
    /// recorded values for inference within the same execution.
    pub fn read(&self, graph: &StateGraph) -> Result<(Tensor, Tensor)> {
        Ok((graph.read(&self.mean)?, graph.read(&self.variance)?))
    }
    /// Record EMA: `(1-rate)*old + rate*batch`, using detached batch mean and
    /// population variance (no unbiased correction). Rate is a finite construction
    /// time number in [0,1], explicitly weighting the NEW batch. Rate 0 preserves
    /// old values and rate 1 replaces them without 0*NaN contamination.
    /// Runtime statistic values are not validated or clamped.
    pub fn update(
        &self,
        graph: &mut StateGraph,
        batch: &BatchNormTraining,
        rate: f32,
    ) -> Result<()> {
        self.prepare(graph, batch, rate)?.commit(graph)
    }
    /// Conditionally commit BOTH statistics. Zero mask keeps both old versions;
    /// nonzero (including NaN) selects both new versions. The scalar graph-local
    /// mask should be an explicit predicate, not a raw loss. Computation/output
    /// is not skipped or guarded; parameters, optimizer and RNG are not included
    /// in this commit group. Errors leave both symbolic versions unchanged.
    pub fn update_if(
        &self,
        graph: &mut StateGraph,
        batch: &BatchNormTraining,
        rate: f32,
        condition: &Tensor,
    ) -> Result<()> {
        self.prepare(graph, batch, rate)?
            .commit_if(graph, condition)
    }
    /// Build detached EMA proposals without recording state writes. Use
    /// finite_mask() to include statistic finiteness in a shared training guard,
    /// then commit_if() with the final acceptance predicate. Rate and shape rules
    /// are identical to update(); finite checking is opt-in, not implicit.
    pub fn prepare(
        &self,
        graph: &StateGraph,
        batch: &BatchNormTraining,
        rate: f32,
    ) -> Result<BatchNormUpdate> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(err("BatchNorm update rate must be finite and in [0,1]"));
        }
        let (mean, variance) = self.read(graph)?;
        for value in [&batch.mean, &batch.variance] {
            if value.shape() != [self.channels as i64] {
                return Err(err("BatchNorm statistics shape mismatch"));
            }
            if !Arc::ptr_eq(&mean.graph().0, &value.graph().0) {
                return Err(err("cross-graph BatchNorm statistics"));
            }
        }
        let blend = |old: &Tensor, new: &Tensor| {
            if rate == 0. {
                Ok(old.clone())
            } else if rate == 1. {
                new.detach()
            } else {
                old.mul_scalar(1. - rate)?
                    .add(&new.detach()?.mul_scalar(rate)?)
            }
        };
        let proposed = [
            blend(&mean, &batch.mean)?,
            blend(&variance, &batch.variance)?,
        ];
        Ok(BatchNormUpdate {
            state: self.clone(),
            original: [mean, variance],
            proposed,
        })
    }
}

impl BatchNormUpdate {
    /// Scalar F32 0/1: every proposed mean/variance element is finite. Does not
    /// check nonnegative variance, loss, gradients or other state. Rate 0 checks
    /// retained statistics, not unused batch values; rate 1 checks replacements.
    /// Merely constructing this predicate does not change commit behavior.
    pub fn finite_mask(&self) -> Result<Tensor> {
        self.proposed[0]
            .is_finite_mask()?
            .min(&[0], false)?
            .mul(&self.proposed[1].is_finite_mask()?.min(&[0], false)?)
    }

    /// Commit without a finite guard, preserving update() semantics.
    pub fn commit(self, graph: &mut StateGraph) -> Result<()> {
        self.record(graph, None)
    }

    /// Commit both proposals on a nonzero scalar predicate (NaN counts as true).
    /// Combine finite_mask() explicitly when desired. Consumes the proposal;
    /// stale/foreign slots or invalid conditions leave both versions unchanged.
    /// Other graph-building calls are not part of this commit group.
    pub fn commit_if(self, graph: &mut StateGraph, condition: &Tensor) -> Result<()> {
        self.record(graph, Some(condition))
    }

    fn record(self, graph: &mut StateGraph, condition: Option<&Tensor>) -> Result<()> {
        let slots = [&self.state.mean, &self.state.variance];
        for (slot, original) in slots.iter().zip(&self.original) {
            if graph.read(slot)?.node_id() != original.node_id() {
                return Err(err(
                    "stale BatchNorm update: statistics changed since prepare",
                ));
            }
        }
        let updates = [
            (slots[0], self.proposed[0].clone()),
            (slots[1], self.proposed[1].clone()),
        ];
        match condition {
            Some(condition) => graph.write_many_if(condition, &updates),
            None => graph.record_updates(&updates),
        }
    }
}
