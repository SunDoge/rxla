//! Weight-only device residency budgeting for serving pipelines.

use crate::{Error, OffloadedSchemaBuffers, PrefetchedSchemaBuffers, Result, SchemaBuffers};
use rxla_nn::{BoundModel, CompiledModel, ModelInputValues, ModelOutputValues};
use rxla_pjrt::{Buffer, Client, Memory};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Budget {
    limit: usize,
    reserved: AtomicUsize,
}

impl Budget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        let result = self
            .reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                reserved
                    .checked_add(bytes)
                    .filter(|&next| next <= self.limit)
            });
        match result {
            Ok(_) => Ok(Reservation {
                budget: self.clone(),
                bytes,
            }),
            Err(reserved) => Err(Error::ResidencyBudget {
                requested: bytes,
                available: self.limit.saturating_sub(reserved),
            }),
        }
    }
}

struct Reservation {
    budget: Arc<Budget>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Shared weight budget for an execution memory space.
///
/// The limit covers only logical parameter/state payloads prefetched through
/// this pool. Reserve room separately for activations, executable temporaries,
/// allocator fragmentation, and buffers created outside the pool.
#[derive(Clone)]
pub struct WeightResidencyPool {
    client: Client,
    execution_memory: Memory,
    budget: Arc<Budget>,
    allocator_headroom_bytes: Option<usize>,
}

impl WeightResidencyPool {
    pub fn new(execution_memory: Memory, weight_budget_bytes: usize) -> Self {
        Self {
            client: execution_memory.client(),
            execution_memory,
            budget: Arc::new(Budget {
                limit: weight_budget_bytes,
                reserved: AtomicUsize::new(0),
            }),
            allocator_headroom_bytes: None,
        }
    }

    /// Require this many allocator bytes to remain available after each
    /// prefetch. This is an advisory point-in-time guard for activations and
    /// executable temporaries; the shared logical weight budget remains the
    /// concurrency-safe authority for allocations made through this pool.
    pub fn with_allocator_headroom(mut self, bytes: usize) -> Self {
        self.allocator_headroom_bytes = Some(bytes);
        self
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget.limit
    }

    pub fn reserved_bytes(&self) -> usize {
        self.budget.reserved.load(Ordering::Acquire)
    }

    pub fn available_bytes(&self) -> usize {
        self.budget_bytes().saturating_sub(self.reserved_bytes())
    }

    /// Reserve budget and enqueue an asynchronous prefetch.
    ///
    /// The reservation is rolled back automatically if submission fails and is
    /// released when the returned resident bundle is dropped.
    pub fn prefetch<'pool, 'offload>(
        &'pool self,
        source: &'offload OffloadedSchemaBuffers,
    ) -> Result<ResidentSchemaBuffers<'pool, 'offload>> {
        if let Some(headroom) = self.allocator_headroom_bytes {
            let stats = self
                .client
                .memory_stats()
                .map_err(|source| Error::Runtime { source })?;
            if let Some(limit) = stats
                .bytes_limit
                .and_then(|bytes| usize::try_from(bytes).ok())
            {
                let in_use = usize::try_from(stats.bytes_in_use).unwrap_or(usize::MAX);
                let available = limit.saturating_sub(in_use);
                let requested = source.payload_bytes().checked_add(headroom).ok_or(
                    Error::DeviceMemoryPressure {
                        requested: usize::MAX,
                        available,
                    },
                )?;
                if requested > available {
                    return Err(Error::DeviceMemoryPressure {
                        requested,
                        available,
                    });
                }
            }
        }
        let reservation = self.budget.reserve(source.payload_bytes())?;
        let buffers = source.prefetch_to(&self.execution_memory)?;
        Ok(ResidentSchemaBuffers {
            buffers,
            _reservation: reservation,
            _pool: std::marker::PhantomData,
        })
    }
}

/// Prefetched weights holding a reservation in their originating pool.
pub struct ResidentSchemaBuffers<'pool, 'offload> {
    buffers: PrefetchedSchemaBuffers<'offload>,
    _reservation: Reservation,
    _pool: std::marker::PhantomData<&'pool WeightResidencyPool>,
}

impl<'pool, 'offload> ResidentSchemaBuffers<'pool, 'offload> {
    pub fn buffers(&self) -> &SchemaBuffers {
        self.buffers.buffers()
    }

    pub fn bindings(&self) -> Vec<(&str, &Buffer)> {
        self.buffers.bindings()
    }

    pub fn state_bindings(&self) -> Vec<(&str, &Buffer)> {
        self.buffers.state_bindings()
    }

    pub fn payload_bytes(&self) -> usize {
        self.buffers.payload_bytes()
    }

    pub fn block_until_ready(&self) -> Result<()> {
        self.buffers.block_until_ready()
    }

    /// Bind weights while transferring the residency lease into the runner.
    ///
    /// `BoundModel` clones PJRT buffer handles. Keeping this wrapper alongside
    /// it prevents the pool from reporting the weight bytes as free while the
    /// runner still owns those cloned handles.
    pub fn bind(
        self,
        compiled: &CompiledModel,
    ) -> std::result::Result<ResidentBoundModel<'pool, 'offload>, rxla_nn::Error> {
        let model = compiled.bind_parameters(self.bindings())?;
        Ok(ResidentBoundModel {
            model,
            _weights: self,
        })
    }
}

/// A bound runner that keeps its weight-budget reservation alive.
pub struct ResidentBoundModel<'pool, 'offload> {
    model: BoundModel,
    _weights: ResidentSchemaBuffers<'pool, 'offload>,
}

impl ResidentBoundModel<'_, '_> {
    pub fn run<'inputs, I, O>(&self, inputs: I) -> std::result::Result<O, rxla_nn::Error>
    where
        I: ModelInputValues<'inputs>,
        O: ModelOutputValues,
    {
        self.model.run(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_atomic_bounded_and_released() {
        let budget = Arc::new(Budget {
            limit: 10,
            reserved: AtomicUsize::new(0),
        });
        let first = budget.reserve(7).unwrap();
        assert_eq!(budget.reserved.load(Ordering::Acquire), 7);
        assert!(matches!(
            budget.reserve(4),
            Err(Error::ResidencyBudget {
                requested: 4,
                available: 3
            })
        ));
        drop(first);
        assert_eq!(budget.reserved.load(Ordering::Acquire), 0);
        let entire_budget = budget.reserve(10).unwrap();
        assert_eq!(budget.reserved.load(Ordering::Acquire), 10);
        drop(entire_budget);
    }
}
