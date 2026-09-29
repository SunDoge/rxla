//! Weight-only device residency budgeting for serving pipelines.

use crate::{Error, OffloadedSchemaBuffers, PrefetchedSchemaBuffers, Result, SchemaBuffers};
use rxla_pjrt::{Buffer, Memory};
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
    execution_memory: Memory,
    budget: Arc<Budget>,
}

impl WeightResidencyPool {
    pub fn new(execution_memory: Memory, weight_budget_bytes: usize) -> Self {
        Self {
            execution_memory,
            budget: Arc::new(Budget {
                limit: weight_budget_bytes,
                reserved: AtomicUsize::new(0),
            }),
        }
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

impl ResidentSchemaBuffers<'_, '_> {
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
