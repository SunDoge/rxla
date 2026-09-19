//! Ordered bounded host pipelines for input preparation.

use snafu::{Snafu, ensure};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Condvar, Mutex,
    mpsc::{self, Receiver},
};
use std::thread::{self, JoinHandle};

use crate::DataRng;

struct PipelineCredits {
    state: Mutex<(usize, bool)>,
    available: Condvar,
}

impl PipelineCredits {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new((capacity, false)),
            available: Condvar::new(),
        }
    }

    fn acquire(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while state.0 == 0 && !state.1 {
            state = self
                .available
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if state.1 {
            return false;
        }
        state.0 -= 1;
        true
    }

    fn release(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.0 += 1;
        self.available.notify_one();
    }

    fn close(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.1 = true;
        self.available.notify_all();
    }
}

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum PipelineError {
    #[snafu(display("pipeline channel capacity must be nonzero"))]
    ZeroCapacity,
    #[snafu(display("shuffle pool capacity must be nonzero"))]
    ZeroShuffleCapacity,
    #[snafu(display("parallel pipeline worker count must be nonzero"))]
    ZeroWorkers,
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
/// Sequential stages run on one OS thread; explicitly parallel stages use their
/// requested bounded worker count. Capacity bounds the number of values retained
/// between adjacent stages and therefore provides backpressure. Values remain
/// ordered, and the first item error terminates downstream processing.
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

    /// Append a bounded parallel transformation while preserving input order.
    ///
    /// Workers claim monotonically numbered inputs from one synchronized source.
    /// The collector buffers only out-of-order completions and emits them in
    /// source order, so scheduling never changes deterministic data pipelines.
    /// The first ordered error terminates downstream processing.
    pub fn map_parallel_ordered<U, F>(
        self,
        capacity: usize,
        worker_count: usize,
        transform: F,
    ) -> PipelineResult<BoundedPipeline<U, E>>
    where
        U: Send + 'static,
        F: Fn(T) -> Result<U, E> + Send + Sync + 'static,
    {
        ensure!(capacity != 0, ZeroCapacitySnafu);
        ensure!(worker_count != 0, ZeroWorkersSnafu);
        let Self {
            receiver,
            mut workers,
        } = self;
        let transform = Arc::new(transform);
        let input = Arc::new(Mutex::new((receiver, 0_usize)));
        let (completed_sender, completed_receiver) =
            mpsc::sync_channel::<(usize, Result<U, E>)>(capacity);
        // Credits bound the whole parallel stage, including items waiting in
        // the reorder map. A closeable semaphore also wakes idle workers when
        // the input ends or downstream abandons the pipeline.
        let credits = Arc::new(PipelineCredits::new(capacity));
        for _ in 0..worker_count {
            let input = input.clone();
            let credits = credits.clone();
            let completed_sender = completed_sender.clone();
            let transform = transform.clone();
            workers.push(thread::spawn(move || {
                loop {
                    if !credits.acquire() {
                        break;
                    }
                    let mut input = input
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let Some(value) = input.0.recv().ok() else {
                        credits.close();
                        break;
                    };
                    let index = input.1;
                    input.1 += 1;
                    drop(input);
                    let value = value.and_then(|value| transform(value));
                    let failed = value.is_err();
                    if completed_sender.send((index, value)).is_err() {
                        credits.close();
                        break;
                    }
                    if failed {
                        credits.close();
                        break;
                    }
                }
            }));
        }
        drop(completed_sender);

        let (sender, next_receiver) = mpsc::sync_channel(capacity);
        let collector_credits = credits.clone();
        workers.push(thread::spawn(move || {
            let mut expected = 0;
            let mut pending = BTreeMap::new();
            for (index, value) in completed_receiver {
                pending.insert(index, value);
                while let Some(value) = pending.remove(&expected) {
                    let failed = value.is_err();
                    if sender.send(value).is_err() || failed {
                        collector_credits.close();
                        return;
                    }
                    collector_credits.release();
                    expected += 1;
                }
            }
            collector_credits.close();
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

        let pipeline = BoundedPipeline::from_iter(1, (0..1_000_000).map(Ok::<_, ()>))
            .unwrap()
            .map_parallel_ordered(1, 4, |value| Ok(value + 1))
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
        let pipeline =
            BoundedPipeline::from_iter(1, std::iter::empty::<Result<i32, ()>>()).unwrap();
        assert!(matches!(
            pipeline.map_parallel_ordered(1, 0, Ok),
            Err(PipelineError::ZeroWorkers)
        ));
    }

    #[test]
    fn parallel_stage_restores_order_before_propagating_errors() {
        use std::time::Duration;

        let values = (0..12).map(|value| if value == 9 { Err(value) } else { Ok(value) });
        let pipeline = BoundedPipeline::from_iter(3, values)
            .unwrap()
            .map_parallel_ordered(3, 4, |value| {
                thread::sleep(Duration::from_millis((3 - value % 4) as u64));
                Ok(value * value)
            })
            .unwrap();
        let mut output = Vec::new();
        while let Some(value) = pipeline.recv() {
            output.push(value);
        }
        pipeline.finish().unwrap();
        assert_eq!(
            output,
            [
                Ok(0),
                Ok(1),
                Ok(4),
                Ok(9),
                Ok(16),
                Ok(25),
                Ok(36),
                Ok(49),
                Ok(64),
                Err(9)
            ]
        );
    }

    #[test]
    fn parallel_stage_capacity_bounds_out_of_order_work() {
        use std::sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        };
        use std::time::{Duration, Instant};

        let first_started = Arc::new(Barrier::new(2));
        let release_first = Arc::new(Barrier::new(2));
        let started = Arc::new(AtomicUsize::new(0));
        let pipeline = BoundedPipeline::from_iter(2, (0..20).map(Ok::<_, ()>))
            .unwrap()
            .map_parallel_ordered(2, 4, {
                let first_started = first_started.clone();
                let release_first = release_first.clone();
                let started = started.clone();
                move |value| {
                    started.fetch_add(1, Ordering::SeqCst);
                    if value == 0 {
                        first_started.wait();
                        release_first.wait();
                    }
                    Ok(value)
                }
            })
            .unwrap();
        first_started.wait();
        let deadline = Instant::now() + Duration::from_secs(1);
        while started.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(started.load(Ordering::SeqCst), 2);
        release_first.wait();
        let mut output = Vec::new();
        while let Some(value) = pipeline.recv() {
            output.push(value.unwrap());
        }
        pipeline.finish().unwrap();
        assert_eq!(output, (0..20).collect::<Vec<_>>());
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
