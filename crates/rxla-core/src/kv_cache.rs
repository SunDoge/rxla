//! Paired, fixed-capacity F32 inference state.
use super::*;

/// Symbolic K/V state belonging to one `StateGraph`, not live device buffers.
/// Each session owns its own contents. This handle is intentionally not Clone;
/// slot accessors identify state for initialization/inspection, not copied data.
pub struct KvCache {
    keys: StateSlot,
    values: StateSlot,
}

/// Results of a bounds-checked paired update already recorded in StateGraph.
pub struct KvCacheUpdate {
    pub keys: Tensor,
    pub values: Tensor,
    /// Scalar F32 1 for valid runtime starts, 0 when both caches stay unchanged.
    /// Use it to gate any separately managed position/length state explicitly.
    pub accepted: Tensor,
}

impl KvCache {
    /// Register two equally shaped F32 slots. Layout is chosen by the caller.
    pub fn new(graph: &mut StateGraph, shape: &[i64]) -> Result<Self> {
        Ok(Self {
            keys: graph.state(shape)?,
            values: graph.state(shape)?,
        })
    }

    pub fn key_slot(&self) -> &StateSlot {
        &self.keys
    }

    pub fn value_slot(&self) -> &StateSlot {
        &self.values
    }

    /// Read the latest symbolic versions, including earlier recorded updates.
    pub fn read(&self, graph: &StateGraph) -> Result<(Tensor, Tensor)> {
        Ok((graph.read(&self.keys)?, graph.read(&self.values)?))
    }

    /// Record a paired dynamic-slice update and return the full updated caches.
    /// K/V updates must have equal shapes. Starts are one graph-local scalar I32
    /// index per axis. As with `Tensor::dynamic_update_slice`, out-of-range starts
    /// are clamped, NOT rejected; this is not append, a ring buffer, or a length
    /// counter. The caller supplies valid positions and an attention mask.
    ///
    /// Neither slot advances if validation fails, though temporary graph nodes
    /// may have been built. Device state changes only when a session executes
    /// successfully. `&mut self` expresses recording intent, not native buffer
    /// uniqueness; the existing non-donating session commit contract applies.
    pub fn update_at(
        &mut self,
        graph: &mut StateGraph,
        keys: &Tensor,
        values: &Tensor,
        starts: &[Tensor],
    ) -> Result<(Tensor, Tensor)> {
        if keys.shape() != values.shape() {
            return Err(err("K/V updates must have equal shapes"));
        }
        let (old_keys, old_values) = self.read(graph)?;
        let keys = old_keys.dynamic_update_slice(keys, starts)?;
        let values = old_values.dynamic_update_slice(values, starts)?;
        graph.write_many(&[(&self.keys, &keys), (&self.values, &values)])?;
        Ok((keys, values))
    }

    /// Record a paired update only when every runtime start fits without clamping.
    /// Invalid starts preserve BOTH caches and return accepted=0; valid starts
    /// return accepted=1. This checks bounds, not append ordering, finiteness or
    /// cache validity. Overwrites of an in-bounds region are allowed. Position
    /// counters and attention masks remain explicit caller responsibilities.
    ///
    /// Static shape/ownership errors return Err before either symbolic slot
    /// advances. Rejected runtime updates still build/evaluate graph operands;
    /// selection is not lazy execution. No native buffer donation is introduced.
    pub fn update_at_checked(
        &mut self,
        graph: &mut StateGraph,
        keys: &Tensor,
        values: &Tensor,
        starts: &[Tensor],
    ) -> Result<KvCacheUpdate> {
        if keys.shape() != values.shape() {
            return Err(err("K/V updates must have equal shapes"));
        }
        let (old_keys, old_values) = self.read(graph)?;
        let proposed_keys = old_keys.dynamic_update_slice(keys, starts)?;
        let proposed_values = old_values.dynamic_update_slice(values, starts)?;
        let zero = graph.scalar_i32(0)?;
        let mut accepted = graph.constant(&[], &[1.])?;
        for (axis, start) in starts.iter().enumerate() {
            accepted = accepted.mul(&zero.le_mask(start)?)?;
            let limit = old_keys.shape()[axis] - keys.shape()[axis];
            if limit < i64::from(i32::MAX) {
                accepted = accepted.mul(&start.le_mask(&graph.scalar_i32(limit as i32)?)?)?;
            }
        }
        let accepted = accepted.detach()?;
        let mask = accepted.broadcast_to(old_keys.shape())?;
        let keys = mask.select(&proposed_keys, &old_keys)?;
        let values = mask.select(&proposed_values, &old_values)?;
        graph.write_many(&[(&self.keys, &keys), (&self.values, &values)])?;
        Ok(KvCacheUpdate {
            keys,
            values,
            accepted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_pair_preserves_both_versions() {
        let mut g = StateGraph::default();
        let mut cache = KvCache::new(&mut g, &[4, 2]).unwrap();
        let (old_k, old_v) = cache.read(&g).unwrap();
        let good = g.input(&[1, 2]).unwrap();
        let wrong = g.input(&[1, 1]).unwrap();
        let oversized = g.input(&[5, 2]).unwrap();
        let starts = [g.scalar_i32(1).unwrap(), g.scalar_i32(0).unwrap()];
        let mut other = StateGraph::default();
        let foreign = other.input(&[1, 2]).unwrap();
        let foreign_start = other.scalar_i32(0).unwrap();
        for (k, v, indices) in [
            (&good, &wrong, starts.as_slice()),
            (&good, &foreign, starts.as_slice()),
            (&oversized, &oversized, starts.as_slice()),
            (&good, &good, &starts[..1]),
            (&good, &good, &[foreign_start, starts[1].clone()]),
        ] {
            assert!(cache.update_at(&mut g, k, v, indices).is_err());
            assert!(cache.update_at_checked(&mut g, k, v, indices).is_err());
            let (k, v) = cache.read(&g).unwrap();
            assert_eq!(
                (k.node_id(), v.node_id()),
                (old_k.node_id(), old_v.node_id())
            );
        }
        assert!(cache.read(&other).is_err());
        assert!(
            cache
                .update_at(&mut other, &foreign, &foreign, &starts)
                .is_err()
        );
        let (k, v) = cache.update_at(&mut g, &good, &good, &starts).unwrap();
        let (actual_k, actual_v) = cache.read(&g).unwrap();
        assert_eq!(actual_k.shape(), k.shape());
        assert_eq!(actual_v.shape(), v.shape());
        assert_ne!(actual_k.node_id(), old_k.node_id());
        assert_ne!(actual_v.node_id(), old_v.node_id());
        assert_ne!(k.node_id(), old_k.node_id());
        assert_ne!(v.node_id(), old_v.node_id());
    }
}
