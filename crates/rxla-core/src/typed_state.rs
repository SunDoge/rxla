//! Lightweight state dtype checking; shapes and graph identities remain dynamic.
use crate::{Result, StateGraph, StateSlot, Tensor, err};
use std::marker::PhantomData;

/// Marker for F32 state, read/written as Tensor.
#[derive(Clone, Copy, Debug)]
pub struct F32;
/// Marker for exact I32 state, read/written as Tensor.
#[derive(Clone, Copy, Debug)]
pub struct I32;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::F32 {}
    impl Sealed for super::I32 {}
}

/// Sealed mapping from supported state markers to symbolic value types.
pub trait StateDType: sealed::Sealed {
    const DTYPE: crate::DType;
    type Value: Clone + Into<Tensor>;
    #[doc(hidden)]
    fn read(graph: &StateGraph, slot: &StateSlot) -> Result<Self::Value>;
}
impl StateDType for F32 {
    const DTYPE: crate::DType = crate::DType::F32;
    type Value = Tensor;
    fn read(graph: &StateGraph, slot: &StateSlot) -> Result<Tensor> {
        graph.read(slot)
    }
}
impl StateDType for I32 {
    const DTYPE: crate::DType = crate::DType::I32;
    type Value = Tensor;
    fn read(graph: &StateGraph, slot: &StateSlot) -> Result<Tensor> {
        graph.read(slot)
    }
}

/// Exclusive symbolic state-update scope. Reads see versions at entry, never
/// staged proposals. No mutable graph access is exposed. Drop discards proposals;
/// commit consumes the scope. This does not roll back tensor nodes or host effects
/// and is not a transaction on running device work.
///
/// ```compile_fail
/// use rxla_core::{F32, State, StateGraph};
/// let mut graph = StateGraph::default();
/// let state = State::<F32>::new(&mut graph, &[]).unwrap();
/// let value = graph.constant(&[], &[1.]).unwrap();
/// let tx = graph.transaction();
/// state.write(&mut graph, &value).unwrap(); // exclusive borrow is still live
/// tx.commit().unwrap();
/// ```
/// ```compile_fail
/// use rxla_core::StateGraph;
/// let mut graph = StateGraph::default();
/// let tx = graph.transaction();
/// tx.commit().unwrap();
/// tx.commit().unwrap(); // consumed by the first commit
/// ```
#[must_use = "commit explicitly; dropping discards staged updates"]
pub struct StateTransaction<'a> {
    graph: &'a mut StateGraph,
    updates: StateUpdates,
}
impl StateGraph {
    pub fn transaction(&mut self) -> StateTransaction<'_> {
        StateTransaction {
            graph: self,
            updates: StateUpdates::new(),
        }
    }
}
impl StateTransaction<'_> {
    /// Read the entry version. Build tensor expressions from this value; queued
    /// writes do not affect subsequent reads in this scope.
    pub fn read<T: StateDType>(&self, state: &State<T>) -> Result<T::Value> {
        T::read(self.graph, state.as_slot())
    }
    /// Stage and validate a typed value. Errors discard only this proposal;
    /// earlier valid proposals remain staged, without advancing any slot.
    pub fn set<T: StateDType>(&mut self, state: &State<T>, value: &T::Value) -> Result<&mut Self> {
        self.updates.set(state, value);
        let values: Vec<_> = self
            .updates
            .updates
            .iter()
            .map(|(s, v)| (s, v.clone()))
            .collect();
        if let Err(error) = self.graph.validate_updates(&values) {
            self.updates.updates.pop();
            return Err(error);
        }
        Ok(self)
    }
    /// Owned chaining variant. An error drops the scope and every staged
    /// proposal, because ownership was passed into this call.
    pub fn with<T: StateDType>(mut self, state: &State<T>, value: &T::Value) -> Result<Self> {
        self.set(state, value)?;
        Ok(self)
    }
    pub fn commit(self) -> Result<()> {
        self.updates.commit(self.graph)
    }
    /// Zero rejects all proposals; nonzero (including NaN) accepts all. Both
    /// branches may compute. Invalid conditions advance no symbolic state.
    pub fn commit_if(self, condition: &Tensor) -> Result<()> {
        self.updates.commit_if(self.graph, condition)
    }
}

/// An owned collection of proposed symbolic updates. `set` only collects values;
/// commit validates the entire group before advancing any slot. Dropping the
/// group discards proposals without committing. Already-built graph nodes and
/// arbitrary Rust effects are not rolled back. This is not a device transaction.
///
/// ```
/// use rxla_core::{State, I32, StateGraph, StateUpdates};
/// let mut graph = StateGraph::default();
/// let counter = State::<I32>::new(&mut graph, &[]).unwrap();
/// let float = graph.constant(&[], &[1.]).unwrap();
/// let mut updates = StateUpdates::new();
/// updates.set(&counter, &float);
/// assert!(updates.commit(&mut graph).is_err());
/// ```
#[derive(Default)]
#[must_use = "commit the group explicitly; dropping discards proposals"]
pub struct StateUpdates {
    updates: Vec<(StateSlot, Tensor)>,
}
impl StateUpdates {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage a proposal and return the owned group for chaining into commit.
    /// Moves the builder, without cloning previously collected proposals. Like
    /// `set`, this captures a symbolic value; validation is deferred to commit.
    ///
    /// ```
    /// use rxla_core::{F32, I32, State, StateGraph, StateUpdates};
    /// let mut graph = StateGraph::default();
    /// let mean = State::<F32>::new(&mut graph, &[2])?;
    /// let steps = State::<I32>::new(&mut graph, &[])?;
    /// let next_mean = mean.read(&graph)?.add_scalar(1.)?;
    /// let next_steps = steps.read(&graph)?.wrapping_add_scalar(1)?;
    /// let accept = graph.constant(&[], &[1.])?;
    /// StateUpdates::new()
    ///     .with(&mean, &next_mean)
    ///     .with(&steps, &next_steps)
    ///     .commit_if(&mut graph, &accept)?;
    /// # Ok::<(), rxla_core::Error>(())
    /// ```
    pub fn with<T: StateDType>(mut self, state: &State<T>, value: &T::Value) -> Self {
        self.set(state, value);
        self
    }

    /// Stage a proposal. Dtype/shape/owner/duplicate validation happens
    /// at commit; repeated slot identities are errors, not last-write-wins.
    /// Captures this symbolic value now, not a closure to reevaluate at commit.
    pub fn set<T: StateDType>(&mut self, state: &State<T>, value: &T::Value) -> &mut Self {
        self.updates
            .push((state.slot.clone(), value.clone().into()));
        self
    }

    /// Consume and apply all proposals together. Validation errors leave all
    /// symbolic slots unchanged; omitted slots keep their current values.
    pub fn commit(self, graph: &mut StateGraph) -> Result<()> {
        let updates: Vec<_> = self.updates.iter().map(|(s, v)| (s, v.clone())).collect();
        graph.record_updates(&updates)
    }

    /// Apply all proposals under one scalar F32 mask. Zero keeps versions
    /// current at commit, nonzero (including NaN) selects proposed values.
    /// Inherits StateGraph's eager dataflow, not lazy branch execution.
    pub fn commit_if(self, graph: &mut StateGraph, condition: &Tensor) -> Result<()> {
        let updates: Vec<_> = self.updates.iter().map(|(s, v)| (s, v.clone())).collect();
        graph.write_many_if(condition, &updates)
    }
}

/// Typed identity of a state slot. Cloning aliases the same slot; it does not
/// copy device state. Only F32/I32 constructors are provided. Shapes and graph
/// ownership are checked at runtime by the existing StateGraph implementation.
/// No device allocation, Rust mutation capture or implicit commit occurs here.
///
/// ```
/// use rxla_core::{State, F32, StateGraph};
/// let mut graph = StateGraph::default();
/// let state = State::<F32>::new(&mut graph, &[]).unwrap();
/// let integer = graph.state_i32(&[]).unwrap();
/// let value = graph.read(&integer).unwrap();
/// assert!(state.write(&mut graph, &value).is_err());
/// ```
/// ```
/// use rxla_core::{State, I32, StateGraph};
/// let mut graph = StateGraph::default();
/// let state = State::<I32>::new(&mut graph, &[]).unwrap();
/// let value = graph.constant(&[], &[1.]).unwrap();
/// assert!(state.write(&mut graph, &value).is_err());
/// ```
#[derive(Clone)]
pub struct State<T> {
    slot: StateSlot,
    marker: PhantomData<T>,
}

impl<T> State<T> {
    /// Borrow the erased slot for existing Session/checkpoint/optimizer APIs.
    /// Writes through those APIs still perform runtime dtype/shape checks.
    pub fn as_slot(&self) -> &StateSlot {
        &self.slot
    }
    pub fn into_slot(self) -> StateSlot {
        self.slot
    }
}

macro_rules! state_type {
    ($marker:ty, $value:ty, $create:ident, $read:ident, $write:ident) => {
        impl State<$marker> {
            /// Register a typed state slot. Initial buffers are supplied when
            /// creating a Session, as with the existing StateSlot API.
            pub fn new(graph: &mut StateGraph, dims: &[i64]) -> Result<Self> {
                Ok(Self {
                    slot: graph.$create(dims)?,
                    marker: PhantomData,
                })
            }
            /// Validate dtype and graph ownership before wrapping an existing
            /// slot. The original slot identity and current version are retained.
            pub fn from_slot(graph: &StateGraph, slot: StateSlot) -> Result<Self> {
                if graph.read(&slot)?.dtype() != <$marker as StateDType>::DTYPE {
                    return Err(err("state marker dtype mismatch"));
                }
                Ok(Self {
                    slot,
                    marker: PhantomData,
                })
            }
            /// Read the current symbolic version, not a device buffer.
            pub fn read(&self, graph: &StateGraph) -> Result<$value> {
                graph.$read(&self.slot)
            }
            /// Record a next symbolic value. Shape/owner checks remain dynamic;
            /// changing live state still requires Session execution.
            pub fn write(&self, graph: &mut StateGraph, value: &$value) -> Result<()> {
                graph.$write(&self.slot, value)
            }
            /// Record a guarded update of this one slot. This inherits the
            /// StateGraph mask semantics: zero rejects, nonzero (even NaN)
            /// accepts. For atomic multi-slot proposals use `write_many_if`.
            pub fn write_if(
                &self,
                graph: &mut StateGraph,
                value: &$value,
                condition: &Tensor,
            ) -> Result<()> {
                graph.write_many_if(condition, &[(&self.slot, value.clone().into())])
            }
        }
    };
}
state_type!(F32, Tensor, state, read, write);
state_type!(I32, Tensor, state_i32, read, write);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transaction_reads_entry_versions_and_rejects_bad_proposals() {
        let mut g = StateGraph::default();
        let a = State::<F32>::new(&mut g, &[]).unwrap();
        let b = State::<F32>::new(&mut g, &[]).unwrap();
        let old_a = a.read(&g).unwrap();
        let old_b = b.read(&g).unwrap();
        let wrong = g.constant(&[1], &[99.]).unwrap();
        let mut tx = g.transaction();
        tx.set(&a, &old_b).unwrap();
        assert_eq!(tx.read(&a).unwrap().node_id(), old_a.node_id());
        assert!(tx.set(&a.clone(), &old_b).is_err());
        assert!(tx.set(&b, &wrong).is_err());
        assert_eq!(tx.read(&b).unwrap().node_id(), old_b.node_id());
        tx.set(&b, &old_a).unwrap();
        tx.commit().unwrap();
        assert_eq!(a.read(&g).unwrap().node_id(), old_b.node_id());
        assert_eq!(b.read(&g).unwrap().node_id(), old_a.node_id());

        // Owned chaining errors drop all proposals, not just the last one.
        assert!(
            g.transaction()
                .with(&a, &old_a)
                .unwrap()
                .with(&b, &wrong)
                .is_err()
        );
        assert_eq!(a.read(&g).unwrap().node_id(), old_b.node_id());
        drop(g.transaction().with(&a, &old_a).unwrap());
        assert_eq!(a.read(&g).unwrap().node_id(), old_b.node_id());
        assert!(
            g.transaction()
                .with(&a, &old_a)
                .unwrap()
                .commit_if(&wrong)
                .is_err()
        );
        assert_eq!(a.read(&g).unwrap().node_id(), old_b.node_id());
        g.transaction().commit().unwrap();
    }

    #[test]
    fn group_validation_and_abandonment_never_partially_advance_slots() {
        let mut g = StateGraph::default();
        let f = State::<F32>::new(&mut g, &[2]).unwrap();
        let i = State::<I32>::new(&mut g, &[]).unwrap();
        let wrong_i = State::<I32>::new(&mut g, &[1]).unwrap().read(&g).unwrap();
        let next = g.constant(&[2], &[4., 5.]).unwrap();
        let old = (f.read(&g).unwrap().node_id(), i.read(&g).unwrap().node_id());
        let unchanged = |g: &StateGraph| {
            assert_eq!(
                (f.read(g).unwrap().node_id(), i.read(g).unwrap().node_id()),
                old
            );
        };
        let mut bad_shape = StateUpdates::new();
        bad_shape.set(&f, &next).set(&i, &wrong_i);
        assert!(bad_shape.commit(&mut g).is_err());
        unchanged(&g);

        let duplicate = StateUpdates::new().with(&f, &next).with(&f.clone(), &next);
        assert!(duplicate.commit(&mut g).is_err());
        unchanged(&g);

        let mut foreign = StateGraph::default();
        let foreign_state = State::<F32>::new(&mut foreign, &[2]).unwrap();
        let foreign_value = foreign_state.read(&foreign).unwrap();
        for use_foreign_slot in [false, true] {
            let mut updates = StateUpdates::new();
            updates.set(&i, &i.read(&g).unwrap());
            if use_foreign_slot {
                updates.set(&foreign_state, &next);
            } else {
                updates.set(&f, &foreign_value);
            }
            assert!(updates.commit(&mut g).is_err());
            unchanged(&g);
        }
        for condition in [
            g.constant(&[1], &[1.]).unwrap(),
            foreign.constant(&[], &[1.]).unwrap(),
        ] {
            let mut updates = StateUpdates::new();
            updates.set(&f, &next);
            assert!(updates.commit_if(&mut g, &condition).is_err());
            unchanged(&g);
        }
        let mut abandoned = StateUpdates::new();
        abandoned.set(&f, &next);
        drop(abandoned);
        unchanged(&g);
        StateUpdates::new().commit(&mut g).unwrap();
        unchanged(&g);
    }

    #[test]
    fn group_captures_values_and_swaps_simultaneously() {
        let mut g = StateGraph::default();
        let a = State::<F32>::new(&mut g, &[]).unwrap();
        let b = State::<F32>::new(&mut g, &[]).unwrap();
        let old_a = a.read(&g).unwrap();
        let old_b = b.read(&g).unwrap();
        let mut swap = StateUpdates::new();
        swap.set(&a, &old_b);
        let swap = swap.with(&b, &old_a);
        let intermediate = g.constant(&[], &[99.]).unwrap();
        a.write(&mut g, &intermediate).unwrap();
        swap.commit(&mut g).unwrap();
        assert_eq!(a.read(&g).unwrap().node_id(), old_b.node_id());
        assert_eq!(b.read(&g).unwrap().node_id(), old_a.node_id());
    }
}
