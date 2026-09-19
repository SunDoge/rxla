//! Ordered bounded host pipelines for input preparation.

use snafu::{Snafu, ensure};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

use crate::DataRng;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum PipelineError {
    #[snafu(display("pipeline channel capacity must be nonzero"))]
    ZeroCapacity,
    #[snafu(display("shuffle pool capacity must be nonzero"))]
    ZeroShuffleCapacity,
    #[snafu(display("pipeline worker panicked"))]
    WorkerPanicked,
}

pub type PipelineResult<T> = std::result::Result<T, PipelineError>;

/// Deterministic bounded shuffle over a sequential input stream.
///
/// The source is read in order. Each output selects one item from the in-memory
/// pool and replaces it with the next source item; after EOF, the remaining
/// pool is drained in random order. Random choices depend on `seed` and output
/// position rather than worker scheduling.
pub struct ShufflePool<I, T, E> {
    input: I,
    pool: Vec<T>,
    capacity: usize,
    rng: DataRng,
    draws: u64,
    initialized: bool,
    exhausted: bool,
    pending_error: Option<E>,
}

impl<I, T, E> ShufflePool<I, T, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    pub fn new(input: I, capacity: usize, seed: u64) -> PipelineResult<Self> {
        ensure!(capacity != 0, ZeroShuffleCapacitySnafu);
        Ok(Self {
            input,
            pool: Vec::with_capacity(capacity),
            capacity,
            rng: DataRng::new(seed),
            draws: 0,
            initialized: false,
            exhausted: false,
            pending_error: None,
        })
    }

    fn fill(&mut self) {
        while self.pool.len() < self.capacity && !self.exhausted {
            match self.input.next() {
                Some(Ok(value)) => self.pool.push(value),
                Some(Err(error)) => {
                    self.pending_error = Some(error);
                    self.exhausted = true;
                }
                None => self.exhausted = true,
            }
        }
        self.initialized = true;
    }

    fn random_index(&mut self) -> usize {
        let word = self.rng.sample(0, self.draws).word(0, 0);
        self.draws += 1;
        word as usize % self.pool.len()
    }
}

impl<I, T, E> Iterator for ShufflePool<I, T, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = Result<T, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.initialized {
            self.fill();
        }
        if let Some(error) = self.pending_error.take() {
            self.pool.clear();
            return Some(Err(error));
        }
        if self.pool.is_empty() {
            return None;
        }

        let index = self.random_index();
        let value = self.pool.swap_remove(index);
        if !self.exhausted {
            match self.input.next() {
                Some(Ok(replacement)) => self.pool.push(replacement),
                Some(Err(error)) => {
                    self.pending_error = Some(error);
                    self.exhausted = true;
                }
                None => self.exhausted = true,
            }
        }
        Some(Ok(value))
    }
}

/// One ordered stream of owned values produced by bounded worker stages.
///
/// Every stage runs on one OS thread. Capacity bounds the number of values
/// retained between adjacent stages and therefore provides backpressure. Values
/// remain ordered, and the first item error terminates downstream processing.
/// Dropping the pipeline closes its final receiver; blocked upstream workers then
/// observe disconnection and exit without requiring cancellation or shared flags.
pub struct BoundedPipeline<T, E> {
    receiver: Receiver<Result<T, E>>,
    workers: Vec<JoinHandle<()>>,
}

impl<T, E> BoundedPipeline<T, E>
where
    T: Send + 'static,
    E: Send + 'static,
{
    /// Move a lazy iterator into a source worker.
    pub fn from_iter<I>(capacity: usize, values: I) -> PipelineResult<Self>
    where
        I: IntoIterator<Item = Result<T, E>> + Send + 'static,
        I::IntoIter: Send,
    {
        ensure!(capacity != 0, ZeroCapacitySnafu);
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let worker = thread::spawn(move || {
            for value in values {
                let failed = value.is_err();
                if sender.send(value).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            receiver,
            workers: vec![worker],
        })
    }

    /// Append one ordered transformation stage.
    pub fn map<U, F>(
        self,
        capacity: usize,
        mut transform: F,
    ) -> PipelineResult<BoundedPipeline<U, E>>
    where
        U: Send + 'static,
        F: FnMut(T) -> Result<U, E> + Send + 'static,
    {
        ensure!(capacity != 0, ZeroCapacitySnafu);
        let Self {
            receiver,
            mut workers,
        } = self;
        let (sender, next_receiver) = mpsc::sync_channel(capacity);
        workers.push(thread::spawn(move || {
            for value in receiver {
                let value = value.and_then(&mut transform);
                let failed = value.is_err();
                if sender.send(value).is_err() || failed {
                    break;
                }
            }
        }));
        Ok(BoundedPipeline {
            receiver: next_receiver,
            workers,
        })
    }

    /// Block until the next item arrives or every producer has exited.
    pub fn recv(&self) -> Option<Result<T, E>> {
        self.receiver.recv().ok()
    }

    /// Close the consumer end and join every worker, including when the pipeline
    /// has not been drained. Workers are joined downstream-first so closure
    /// propagates through bounded channels without deadlock.
    pub fn finish(self) -> PipelineResult<()> {
        let Self {
            receiver,
            mut workers,
        } = self;
        drop(receiver);
        while let Some(worker) = workers.pop() {
            worker.join().map_err(|_| PipelineError::WorkerPanicked)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_preserve_order_and_stop_after_first_item_error() {
        let pipeline = BoundedPipeline::from_iter(1, (0..8).map(Ok::<_, &'static str>))
            .unwrap()
            .map(1, |value| {
                if value == 5 {
                    Err("five")
                } else {
                    Ok(value * value)
                }
            })
            .unwrap();
        let mut values = Vec::new();
        while let Some(value) = pipeline.recv() {
            values.push(value);
        }
        assert_eq!(values, [Ok(0), Ok(1), Ok(4), Ok(9), Ok(16), Err("five")]);
        pipeline.finish().unwrap();
    }

    #[test]
    fn early_finish_unblocks_full_upstream_channels() {
        let pipeline = BoundedPipeline::from_iter(1, (0..1_000_000).map(Ok::<_, ()>))
            .unwrap()
            .map(1, |value| Ok(value + 1))
            .unwrap();
        pipeline.finish().unwrap();
    }

    #[test]
    fn rejects_zero_capacity_and_reports_panics() {
        assert!(matches!(
            BoundedPipeline::<i32, ()>::from_iter(0, std::iter::empty()),
            Err(PipelineError::ZeroCapacity)
        ));
        let pipeline = BoundedPipeline::from_iter(1, std::iter::once(Ok::<_, ()>(1)))
            .unwrap()
            .map(1, |_| -> Result<i32, ()> { panic!("worker failure") })
            .unwrap();
        assert!(pipeline.recv().is_none());
        assert!(matches!(
            pipeline.finish(),
            Err(PipelineError::WorkerPanicked)
        ));
    }

    #[test]
    fn shuffle_pool_is_reproducible_and_preserves_every_item() {
        let shuffled = |seed| {
            ShufflePool::new((0..100).map(Ok::<_, ()>), 11, seed)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        let first = shuffled(42);
        assert_eq!(first, shuffled(42));
        assert_ne!(first, shuffled(43));
        assert_ne!(first, (0..100).collect::<Vec<_>>());
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn shuffle_pool_propagates_source_errors_once() {
        let values = [Ok(1), Ok(2), Err("bad"), Ok(3)];
        let output = ShufflePool::new(values.into_iter(), 4, 7)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(output, [Err("bad")]);
        assert!(matches!(
            ShufflePool::<_, i32, ()>::new(std::iter::empty(), 0, 0),
            Err(PipelineError::ZeroShuffleCapacity)
        ));
    }
}
