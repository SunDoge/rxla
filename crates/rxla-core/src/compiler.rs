use super::*;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Immutable lowered program with its pre-encoded cache key. Lowering requires
/// no plugin and retains no source Graph or native handles. Later graph
/// mutations do not change this snapshot, including its declared input ABI.
/// The backend payload is format-tagged. StableHLO programs retain only their
/// MLIR form; placement is represented by SDY rather than an HLO shadow graph.
#[derive(Clone)]
pub struct LoweredProgram {
    code: Vec<u8>,
    inputs: Vec<TensorType>,
    outputs: Vec<TensorType>,
}

impl LoweredProgram {
    /// PJRT program format used for tensor compilation.
    pub fn format(&self) -> &'static str {
        "mlir"
    }

    /// Serialized source accepted by the backend compiler.
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    pub(crate) fn is_stablehlo(&self) -> bool {
        true
    }

    pub(crate) fn compile_uncached(&self, client: &Client) -> Result<Executable> {
        compile_program_with_options(
            client,
            "mlir",
            &self.code,
            &self.inputs,
            self.outputs.len(),
            None,
        )
    }

    pub(crate) fn cache_key(&self) -> Vec<u8> {
        backend_key("mlir", &self.code)
    }

    /// Number of runtime parameters in this snapshot, with pruning already
    /// applied if requested. Reads host metadata without loading any plugin.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Borrow a parameter's static shape and storage dtype. BF16 inputs report
    /// BF16 even when the graph converts them to F32. No allocation, compilation
    /// or device query occurs; an absent index returns None.
    pub fn input_spec(&self, index: usize) -> Option<InputSpec<'_>> {
        self.inputs.get(index).map(|ty| InputSpec {
            shape: &ty.dims,
            dtype: ty.dtype,
        })
    }

    /// Number of ordered top-level results, including repeated outputs.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Inspect an ordered result without compilation or allocation. Reports
    /// the result dtype, which may differ from the input storage dtype. This
    /// exposes static tensor shapes, not backend physical layouts or buffers.
    pub fn output_spec(&self, index: usize) -> Option<OutputSpec<'_>> {
        self.outputs.get(index).map(|ty| OutputSpec {
            shape: &ty.dims,
            dtype: ty.dtype,
        })
    }

    /// Construct a reusable compiler input from already verified StableHLO.
    /// Frontends should normally obtain the signature from `rxla-ir` rather
    /// than duplicating shape or dtype metadata.
    pub fn from_stablehlo(code: String, inputs: Vec<TensorType>, outputs: Vec<TensorType>) -> Self {
        Self {
            code: code.into_bytes(),
            inputs,
            outputs,
        }
    }
}

impl Graph {
    /// Lower and encode one output once, preserving all declared inputs.
    pub fn prepare(&self, output: &Tensor) -> Result<LoweredProgram> {
        self.prepare_many(std::slice::from_ref(output))
    }

    /// Snapshot ordered F32/I32/BF16 outputs and all currently declared inputs.
    /// Unlike pruned compilation, unused parameters remain in the input ABI.
    /// The snapshot can be reused with different Compiler instances; each keeps
    /// its own device placement, options and memory/disk cache policy.
    pub fn prepare_many(&self, outputs: &[Tensor]) -> Result<LoweredProgram> {
        if outputs.is_empty() {
            return Err(err("program requires at least one output"));
        }
        self.direct_lowered(outputs, true)
    }

    /// Prepare a snapshot with unreachable input parameters removed. The mapping
    /// lists original registration indices in the snapshot's compact input order;
    /// callers must supply buffers in that order, not original registration order.
    /// Output order is unchanged. Neither the source graph nor its AD dependencies
    /// are modified. A constant-only snapshot has an empty input mapping.
    /// This is a value-graph API, not a StateProgram snapshot: state update roots
    /// and slot mappings must not be silently omitted when preparing stateful work.
    pub fn prepare_pruned(&self, outputs: &[Tensor]) -> Result<(LoweredProgram, Vec<usize>)> {
        if outputs.is_empty() {
            return Err(err("program requires at least one output"));
        }
        self.direct_lowered_pruned(outputs)
    }
}

/// Bounds on retained cache entries and serialized backend-program key bytes. Executable
/// memory is backend-owned and is not included in the byte limit.
#[derive(Clone, Copy, Debug)]
pub struct CacheLimits {
    pub max_entries: usize,
    pub max_key_bytes: usize,
}
impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_entries: 32,
            max_key_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub disk_hits: u64,
    pub disk_read_errors: u64,
    /// Serialization/publication failures, including opt-in post-publication trim failures.
    pub disk_write_errors: u64,
    pub hits: u64,
    /// Cache misses that reached the backend, including failed compilations.
    pub misses: u64,
    pub compile_failures: u64,
    /// Cumulative wall time in cache-miss compilation attempts, including failures
    /// and executable metadata setup. Excludes graph lowering, HLO key encoding,
    /// cache reads/writes, cache hits and execution. This is not backend CPU time
    /// or total compile-request latency. `clear` preserves cumulative statistics.
    pub compile_time: Duration,
    pub evictions: u64,
    /// Executables not retained in memory due to size limits or disabled caching.
    pub bypasses: u64,
    pub entries: usize,
    pub key_bytes: usize,
}

/// A synchronous, client-local compiler with an explicit bounded LRU cache.
///
/// The cache key includes the PJRT program format and complete serialized program,
/// including constants, shapes, dtypes and output order. Runtime buffer contents are not keys. Compiler options
/// participate in cache keys when supplied. Separate instances never share
/// in-memory executables, even when
/// their clients use the same plugin. Like PJRT handles, this object is !Send/!Sync.
pub struct Compiler {
    pub(super) client: Client,
    limits: CacheLimits,
    entries: HashMap<Rc<[u8]>, Rc<Executable>>,
    // Keys share storage with the map; oldest access is at the front.
    recency: VecDeque<Rc<[u8]>>,
    stats: CacheStats,
    f16_attention: bool,
    compute_dtype: Option<DType>,
    #[cfg(feature = "disk-cache")]
    disk: Option<DiskCache>,
}
impl Compiler {
    pub(crate) fn lowering_target(&self) -> Result<rxla_ir::LoweringTarget> {
        if !self.client.info()?.platform.eq_ignore_ascii_case("cuda") {
            return Ok(rxla_ir::LoweringTarget::Portable);
        }
        Ok(match self.compute_dtype {
            Some(DType::F16) => rxla_ir::LoweringTarget::CudaF16Compute,
            Some(DType::BF16) => rxla_ir::LoweringTarget::CudaBf16Compute,
            Some(dtype) => {
                return Err(err(format!(
                    "compiler compute dtype must be F16 or BF16, got {dtype:?}"
                )));
            }
            None if self.f16_attention => rxla_ir::LoweringTarget::CudaF16Attention,
            None => rxla_ir::LoweringTarget::Portable,
        })
    }
    pub fn new(client: Client, limits: CacheLimits) -> Self {
        Self {
            client,
            limits,
            entries: HashMap::new(),
            recency: VecDeque::new(),
            stats: CacheStats::default(),
            f16_attention: false,
            compute_dtype: None,
            #[cfg(feature = "disk-cache")]
            disk: None,
        }
    }

    /// Allow eligible long-sequence attention to use F16 cuDNN FMHA on CUDA.
    /// This is an explicit inference precision choice; the default remains F32.
    pub fn with_f16_attention(mut self, enabled: bool) -> Self {
        self.f16_attention = enabled;
        self
    }

    /// Select F16/BF16 internal computation while preserving the F32 public ABI.
    /// CUDA is required; other platforms retain portable F32.
    pub fn with_compute_dtype(mut self, dtype: DType) -> Self {
        self.compute_dtype = Some(dtype);
        self
    }

    /// The client owned by this executor/compiler.
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Borrow the attached cache, retaining its original namespace/flag snapshot.
    /// Inspection/invalidation are explicit and never triggered by this getter.
    #[cfg(feature = "disk-cache")]
    pub fn disk_cache(&self) -> Option<&DiskCache> {
        self.disk.as_ref()
    }

    /// Remove one exact lowered program from memory, leaving disk and other
    /// entries untouched. Previously returned executables remain usable.
    /// Returns false if absent. Preserves hit/miss/automatic-eviction counters;
    /// entries/key_bytes reflect the removal.
    pub fn invalidate_memory(&mut self, program: &LoweredProgram) -> bool {
        let encoded = program.cache_key();
        let Some((key, _)) = self.entries.remove_entry(encoded.as_slice()) else {
            return false;
        };
        let position = self
            .recency
            .iter()
            .position(|k| Rc::ptr_eq(k, &key))
            .expect("cache recency matches entries");
        self.recency.remove(position);
        self.stats.key_bytes -= key.len();
        self.stats.entries = self.entries.len();
        true
    }

    /// Attach a trusted persistent cache. `clear` only clears in-memory entries.
    #[cfg(feature = "disk-cache")]
    pub fn with_disk_cache(mut self, disk: DiskCache) -> Self {
        self.disk = Some(disk);
        self
    }

    /// Drop cache-owned references, preserving counters. Previously returned
    /// executables remain usable until the caller releases them.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.recency.clear();
        self.stats.entries = 0;
        self.stats.key_bytes = 0;
    }

    pub fn compile(&mut self, tracer: &Tracer, output: &Tensor) -> Result<Rc<Executable>> {
        self.compile_many(tracer, std::slice::from_ref(output))
    }

    /// Use a lowered snapshot without repeating graph lowering or program encoding.
    /// Cache keys and executable identity are identical to ordinary compilation.
    /// Lookup still hashes/compares key bytes and updates the LRU; callers that
    /// already own the desired executable should execute that handle directly.
    pub fn compile_lowered(&mut self, program: &LoweredProgram) -> Result<Rc<Executable>> {
        let key = program.cache_key();
        self.compile_encoded(
            &program.code,
            &program.inputs,
            &key,
            program.outputs.len(),
            None,
        )
    }

    pub(crate) fn compile_lowered_with_options(
        &mut self,
        program: &LoweredProgram,
        options: &[u8],
    ) -> Result<Rc<Executable>> {
        let program_key = program.cache_key();
        let mut key = Vec::with_capacity(16 + program_key.len() + options.len());
        key.extend_from_slice(b"rxla-options\0");
        key.extend_from_slice(&(program_key.len() as u64).to_le_bytes());
        key.extend_from_slice(&program_key);
        key.extend_from_slice(options);
        self.compile_encoded(
            &program.code,
            &program.inputs,
            &key,
            program.outputs.len(),
            Some(options),
        )
    }

    pub fn compile_many(&mut self, tracer: &Tracer, outputs: &[Tensor]) -> Result<Rc<Executable>> {
        self.compile_graph_outputs(&tracer.graph, outputs)
    }

    pub(crate) fn compile_graph_outputs(
        &mut self,
        graph: &Graph,
        outputs: &[Tensor],
    ) -> Result<Rc<Executable>> {
        let lowered = graph.direct_lowered_for(outputs, true, self.lowering_target()?)?;
        self.compile_lowered(&lowered)
    }

    pub(crate) fn compile_graph_outputs_pruned(
        &mut self,
        graph: &Graph,
        outputs: &[Tensor],
    ) -> Result<(Rc<Executable>, Vec<usize>)> {
        let (lowered, parameters) = graph.prepare_pruned(outputs)?;
        Ok((self.compile_lowered(&lowered)?, parameters))
    }

    fn compile_encoded(
        &mut self,
        code: &[u8],
        inputs: &[TensorType],
        key_bytes: &[u8],
        output_count: usize,
        compile_options: Option<&[u8]>,
    ) -> Result<Rc<Executable>> {
        let span = tracing::debug_span!(
            "xla.compile",
            outputs = output_count,
            key_bytes = tracing::field::Empty,
            cache_hit = tracing::field::Empty
        );
        let _entered = span.enter();
        // Validation/lowering errors do not count as backend compile attempts.
        span.record("key_bytes", key_bytes.len());
        if let Some((key, executable)) = self.entries.get_key_value(key_bytes) {
            span.record("cache_hit", true);
            let executable = executable.clone();
            let key = key.clone();
            let position = self
                .recency
                .iter()
                .position(|k| Rc::ptr_eq(k, &key))
                .expect("cache recency matches entries");
            self.recency.remove(position);
            self.recency.push_back(key);
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(executable);
        }
        span.record("cache_hit", false);
        #[cfg(feature = "disk-cache")]
        let restored = match self
            .disk
            .as_ref()
            .map(|disk| disk.load(&self.client, key_bytes))
        {
            Some(Ok(Some(executable))) => {
                self.stats.disk_hits = self.stats.disk_hits.saturating_add(1);
                tracing::debug!("persistent compilation cache hit");
                Some(executable)
            }
            Some(Err(error)) => {
                self.stats.disk_read_errors = self.stats.disk_read_errors.saturating_add(1);
                tracing::debug!(%error, "persistent cache read failed; compiling");
                None
            }
            _ => None,
        };
        #[cfg(not(feature = "disk-cache"))]
        let restored: Option<Executable> = None;
        let executable = if let Some(executable) = restored {
            Rc::new(executable)
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            let start = Instant::now();
            let result = compile_program_with_options(
                &self.client,
                "mlir",
                code,
                inputs,
                output_count,
                compile_options,
            );
            self.stats.compile_time = self.stats.compile_time.saturating_add(start.elapsed());
            let executable = match result {
                Ok(executable) => Rc::new(executable),
                Err(error) => {
                    tracing::debug!("backend compilation failed");
                    self.stats.compile_failures = self.stats.compile_failures.saturating_add(1);
                    return Err(error);
                }
            };
            #[cfg(feature = "disk-cache")]
            if let Some(disk) = &self.disk
                && let Err(error) = disk.store(key_bytes, &executable)
            {
                self.stats.disk_write_errors = self.stats.disk_write_errors.saturating_add(1);
                tracing::debug!(%error, "persistent cache write failed");
            }
            executable
        };
        if self.limits.max_entries == 0 || key_bytes.len() > self.limits.max_key_bytes {
            tracing::debug!("compilation cache retention bypassed");
            self.stats.bypasses = self.stats.bypasses.saturating_add(1);
            return Ok(executable);
        }
        // Evict only after a successful compile, so failures leave useful entries intact.
        while self.entries.len() >= self.limits.max_entries
            || self.stats.key_bytes > self.limits.max_key_bytes - key_bytes.len()
        {
            let key = self
                .recency
                .pop_front()
                .expect("nonempty cache over capacity");
            self.entries.remove(&key);
            self.stats.key_bytes -= key.len();
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.stats.key_bytes += key_bytes.len();
        let key: Rc<[u8]> = key_bytes.into();
        self.recency.push_back(key.clone());
        self.entries.insert(key, executable.clone());
        self.stats.entries = self.entries.len();
        Ok(executable)
    }
}

fn backend_key(format: &str, code: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + format.len() + code.len());
    key.extend_from_slice(b"rxla-program\0");
    key.extend_from_slice(format.as_bytes());
    key.push(0);
    key.extend_from_slice(code);
    key
}
