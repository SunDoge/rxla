//! User-facing tracing and execution facade.
//!
//! The mutable graph owner is an implementation detail. Explicit construction
//! uses [`Tracer`], while ordinary code composes lazy tensors. The
//! tracer is Pliron-only and rejects operations that have not been migrated;
//! it never silently switches to the compatibility representation. The
//! runtime owns compilation, placement policy, and PJRT resources while tensor
//! expressions remain runtime-independent until [`Tensor::eval`].

#[cfg(feature = "disk-cache")]
use crate::disk_cache::DiskCache;
use crate::{
    Buffer, CacheLimits, CacheStats, Compiler, Error, Executable, ExecutionPlan, Graph, InputSpec,
    LoweredProgram, OutputSpec, PlanningPolicy, Result, StateGraph, StateProgram, Tensor, err,
};
use prost::Message;
use rxla_pjrt::{Client, ClientOptions, DType};
use rxla_xla_proto::xla::{
    CompileOptionsProto, DeviceAssignmentProto, ExecutableBuildOptionsProto,
    device_assignment_proto::ComputationDevice,
};
use std::{collections::BTreeMap, sync::Arc};

/// A concrete device selected through a PJRT client.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Device {
    backend: String,
    ordinal: usize,
    id: i32,
    kind: String,
}

impl Device {
    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }
}

/// A single IR-building session.
#[derive(Clone, Default)]
pub struct Tracer {
    pub(crate) graph: Graph,
}

impl Tracer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&self, shape: &[i64]) -> Result<Tensor> {
        self.graph.input(shape)
    }

    pub fn input_dtype(&self, shape: &[i64], dtype: DType) -> Result<Tensor> {
        self.graph.input_dtype(shape, dtype)
    }

    pub fn input_bf16_as_f32(&self, shape: &[i64]) -> Result<Tensor> {
        self.graph.input_bf16_as_f32(shape)
    }

    pub fn input_i32(&self, shape: &[i64]) -> Result<Tensor> {
        self.graph.input_i32(shape)
    }

    pub fn input_i32_scalar(&self) -> Result<Tensor> {
        self.graph.input_i32_scalar()
    }

    pub fn constant(&self, shape: &[i64], values: &[f32]) -> Result<Tensor> {
        self.graph.constant(shape, values)
    }

    pub fn constant_i32(&self, shape: &[i64], values: &[i32]) -> Result<Tensor> {
        self.graph.constant_i32(shape, values)
    }

    pub fn scalar_i32(&self, value: i32) -> Result<Tensor> {
        self.graph.scalar_i32(value)
    }

    pub fn iota_i32(&self, shape: &[i64], axis: usize) -> Result<Tensor> {
        self.graph.iota_i32(shape, axis)
    }

    pub fn causal_attention_mask(
        &self,
        queries: i64,
        keys: i64,
        query_offset: i64,
    ) -> Result<Tensor> {
        self.graph
            .causal_attention_mask(queries, keys, query_offset)
    }

    pub fn stablehlo(&self, output: &Tensor) -> Result<String> {
        self.graph.stablehlo(output)
    }

    pub fn stablehlo_many(&self, outputs: &[Tensor]) -> Result<String> {
        self.graph.stablehlo_many(outputs)
    }

    pub fn prepare(&self, output: &Tensor) -> Result<LoweredProgram> {
        self.graph.prepare(output)
    }

    pub fn prepare_many(&self, outputs: &[Tensor]) -> Result<LoweredProgram> {
        self.graph.prepare_many(outputs)
    }

    pub fn prepare_pruned(&self, outputs: &[Tensor]) -> Result<(LoweredProgram, Vec<usize>)> {
        self.graph.prepare_pruned(outputs)
    }

    pub fn compile(&self, client: &Client, output: &Tensor) -> Result<Executable> {
        self.graph.compile(client, output)
    }

    pub fn compile_many(&self, client: &Client, outputs: &[Tensor]) -> Result<Executable> {
        self.graph.compile_many(client, outputs)
    }

    /// Trace a function into a reusable immutable [`Program`].
    pub fn trace<F>(build: F) -> Result<Program>
    where
        F: FnOnce(&Self) -> Result<Vec<Tensor>>,
    {
        let tracer = Self::new();
        let outputs = build(&tracer)?;
        tracer.program(outputs)
    }

    pub fn program(&self, outputs: Vec<Tensor>) -> Result<Program> {
        // Validate roots and graph ownership now, rather than deferring an
        // invalid program until the first backend invocation.
        let output_nodes = outputs.to_vec();
        let (lowered, planning, _, source) = self.graph.direct_program(&output_nodes, false)?;
        Ok(Program {
            planning,
            lowered,
            source,
        })
    }
}

/// An immutable traced computation. It contains no PJRT client or executable.
#[derive(Clone)]
pub struct Program {
    pub(crate) planning: crate::planning::PlanningSnapshot,
    pub(crate) lowered: LoweredProgram,
    source: rxla_ir::SemanticProgram,
}

/// A reusable one-input/one-output Tensor computation.
///
/// The builder receives an ordinary symbolic [`Tensor`]; graph ownership and
/// placeholder creation remain internal. Calling the function accepts either a
/// host-backed or device-resident Tensor, compiles on first use, and reuses the
/// runtime's executable cache on subsequent calls.
#[derive(Clone)]
pub struct TensorFunction {
    program: Program,
}

impl TensorFunction {
    /// Trace a reusable Tensor function for one statically shaped input.
    pub fn new<F>(shape: impl AsRef<[i64]>, dtype: DType, build: F) -> Result<Self>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        let tracer = Tracer::new();
        let input = tracer.input_dtype(shape.as_ref(), dtype)?;
        let output = build(&input)?;
        Ok(Self {
            program: tracer.program(vec![output])?,
        })
    }

    /// Execute with a host-backed or resident Tensor and return a resident
    /// materialized Tensor. Input shape and dtype must match the traced spec.
    pub fn call(&self, runtime: &mut Runtime, input: &Tensor) -> Result<Tensor> {
        self.program
            .run_tensors(runtime, &[input])?
            .into_iter()
            .next()
            .ok_or_else(|| err("TensorFunction execution returned no output"))
    }

    /// Compile without executing. Ordinary callers can rely on first-call
    /// compilation; serving systems may use this during explicit warmup.
    pub fn compile(&self, runtime: &mut Runtime) -> Result<Arc<Executable>> {
        self.program.compile(runtime)
    }

    /// Inspect the reusable program for profiling and deployment tooling.
    pub fn program(&self) -> &Program {
        &self.program
    }
}

impl Program {
    /// Snapshot one or more lazy Tensor expressions as a verified reusable
    /// program without loading PJRT or compiling for a device. This is the fast
    /// path for frontend tests, IR inspection, cache-key preparation, and tools
    /// that do not need runtime execution.
    pub fn from_tensors(outputs: &[Tensor]) -> Result<Self> {
        let first = outputs
            .first()
            .ok_or_else(|| err("program requires at least one Tensor output"))?;
        let graph = first.graph();
        if outputs
            .iter()
            .any(|output| !std::sync::Arc::ptr_eq(&graph.0, &output.graph().0))
        {
            return Err(err("program outputs belong to different lazy traces"));
        }
        let (lowered, planning, _, source) = graph.direct_program(outputs, false)?;
        Ok(Self {
            planning,
            lowered,
            source,
        })
    }

    fn lower_for_compile(&self, target: rxla_ir::LoweringTarget) -> Result<LoweredProgram> {
        let program = self.source.lower(target)?;
        Ok(LoweredProgram::from_stablehlo(
            program.code,
            program.inputs,
            program.outputs,
        ))
    }
}

impl Program {
    pub fn plan(&self, policy: PlanningPolicy) -> Result<ExecutionPlan> {
        self.planning.plan(policy)
    }

    pub fn input_count(&self) -> usize {
        self.lowered.input_count()
    }

    pub fn output_count(&self) -> usize {
        self.lowered.output_count()
    }

    pub fn input_spec(&self, index: usize) -> Option<InputSpec<'_>> {
        self.lowered.input_spec(index)
    }

    pub fn output_spec(&self, index: usize) -> Option<OutputSpec<'_>> {
        self.lowered.output_spec(index)
    }

    /// Inspect the verified backend-neutral compilation artifact.
    pub fn lowered_program(&self) -> &LoweredProgram {
        &self.lowered
    }

    pub fn compile(&self, runtime: &mut Runtime) -> Result<Arc<Executable>> {
        runtime.compile(self)
    }

    pub fn run(&self, runtime: &mut Runtime, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        runtime.run(self, inputs)
    }

    pub fn run_buffers(&self, runtime: &mut Runtime, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        runtime.run_buffers(self, inputs)
    }

    /// Execute managed or materialized Tensor inputs and return materialized
    /// Tensor outputs. Compilation and PJRT dispatch remain executor details.
    pub fn run_tensors(&self, runtime: &mut Runtime, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        runtime.run_tensors(self, inputs)
    }
}

/// Owns PJRT resources, compilation state, and placement policy.
pub struct Runtime {
    backends: BTreeMap<String, Backend>,
    default_device: Device,
    planning_policy: PlanningPolicy,
}

struct Backend {
    compiler: Compiler,
    devices: Vec<Device>,
    default_ordinal: usize,
}

/// Builder for an explicitly owned [`Runtime`].
pub struct RuntimeBuilder {
    backends: Vec<BackendRegistration>,
    default_backend: Option<String>,
    cache_limits: CacheLimits,
    planning_policy: PlanningPolicy,
}

struct BackendRegistration {
    name: String,
    client: Client,
    #[cfg(feature = "disk-cache")]
    disk_cache: Option<DiskCache>,
}

/// A temporary view routing operations to one named backend.
pub struct DeviceRuntime<'a> {
    runtime: &'a mut Runtime,
    device: Device,
}

/// A type-safe collection of lazy values accepted by [`Runtime::eval`].
///
/// Implementations are provided for a single tensor, tensor slices and arrays,
/// and tuples of up to four tensor references. All values in a collection are
/// planned and materialized together so shared computation is preserved.
pub trait Evaluable {
    type Output;

    fn eval_with(self, runtime: &mut Runtime, device: &Device) -> Result<Self::Output>;
}

impl Evaluable for &Tensor {
    type Output = Tensor;

    fn eval_with(self, runtime: &mut Runtime, device: &Device) -> Result<Self::Output> {
        runtime.eval_tensor_on(device, self)
    }
}

impl Evaluable for Tensor {
    type Output = Tensor;

    fn eval_with(self, runtime: &mut Runtime, device: &Device) -> Result<Self::Output> {
        runtime.eval_tensor_on(device, &self)
    }
}

impl Evaluable for &[Tensor] {
    type Output = Vec<Tensor>;

    fn eval_with(self, runtime: &mut Runtime, device: &Device) -> Result<Self::Output> {
        runtime.eval_many_on(device, self)
    }
}

impl<const N: usize> Evaluable for [&Tensor; N] {
    type Output = [Tensor; N];

    fn eval_with(self, runtime: &mut Runtime, device: &Device) -> Result<Self::Output> {
        runtime
            .eval_many_on(device, &self.map(Tensor::clone))?
            .try_into()
            .map_err(|_| err("eval returned an unexpected output count"))
    }
}

macro_rules! impl_tuple_evaluable {
    ($(($($name:ident),+)),+ $(,)?) => {
        $(
            impl<$($name),+> Evaluable for ($(&$name),+)
            where
                $($name: std::borrow::Borrow<Tensor>),+
            {
                type Output = ($(tuple_tensor_type!($name)),+);

                #[allow(non_snake_case)]
                fn eval_with(self, runtime: &mut Runtime, device: &Device) -> Result<Self::Output> {
                    let ($($name),+) = self;
                    let values = vec![$($name.borrow().clone()),+];
                    let mut values = runtime.eval_many_on(device, &values)?.into_iter();
                    Ok(($({
                        let _ = stringify!($name);
                        values.next().expect("tuple arity is preserved")
                    }),+))
                }
            }
        )+
    };
}

macro_rules! tuple_tensor_type {
    ($name:ident) => {
        Tensor
    };
}

impl_tuple_evaluable!((A, B), (A, B, C), (A, B, C, D));

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            backends: Vec::new(),
            default_backend: None,
            cache_limits: CacheLimits::default(),
            planning_policy: PlanningPolicy::SingleDevice,
        }
    }
}

impl RuntimeBuilder {
    pub fn client(mut self, client: Client) -> Self {
        self.backends.push(BackendRegistration {
            name: "default".into(),
            client,
            #[cfg(feature = "disk-cache")]
            disk_cache: None,
        });
        self.default_backend = Some("default".into());
        self
    }

    /// Set the default backend and attach a trusted persistent executable cache.
    ///
    /// # Safety
    /// `directory`, `compatibility_key`, and the client must satisfy
    /// [`DiskCache::new_for_client`]'s native-code trust and compatibility
    /// contract for the lifetime of the resulting runtime.
    #[cfg(feature = "disk-cache")]
    pub unsafe fn cached_client(
        self,
        client: Client,
        directory: impl AsRef<std::path::Path>,
        compatibility_key: &str,
        max_entry_bytes: usize,
    ) -> Result<Self> {
        Ok(unsafe {
            self.cached_backend(
                "default",
                client,
                directory,
                compatibility_key,
                max_entry_bytes,
            )
        }?
        .default_backend("default"))
    }

    /// Add a named PJRT client. CPU and accelerator backends normally use
    /// distinct clients even when they coexist in one runtime.
    pub fn backend(mut self, name: impl Into<String>, client: Client) -> Self {
        self.backends.push(BackendRegistration {
            name: name.into(),
            client,
            #[cfg(feature = "disk-cache")]
            disk_cache: None,
        });
        self
    }

    /// Add a named backend with a trusted persistent executable cache.
    ///
    /// Cache compatibility is backend-scoped: heterogeneous CPU/CUDA runtimes
    /// should use distinct compatibility keys even when they share a directory.
    ///
    /// # Safety
    /// `directory`, `compatibility_key`, and the client must satisfy
    /// [`DiskCache::new_for_client`]'s native-code trust and compatibility
    /// contract for the lifetime of the resulting runtime.
    #[cfg(feature = "disk-cache")]
    pub unsafe fn cached_backend(
        mut self,
        name: impl Into<String>,
        client: Client,
        directory: impl AsRef<std::path::Path>,
        compatibility_key: &str,
        max_entry_bytes: usize,
    ) -> Result<Self> {
        let disk_cache = unsafe {
            DiskCache::new_for_client(directory, compatibility_key, &client, max_entry_bytes)
        }?;
        self.backends.push(BackendRegistration {
            name: name.into(),
            client,
            disk_cache: Some(disk_cache),
        });
        Ok(self)
    }

    pub fn default_backend(mut self, name: impl Into<String>) -> Self {
        self.default_backend = Some(name.into());
        self
    }

    pub fn cache_limits(mut self, limits: CacheLimits) -> Self {
        self.cache_limits = limits;
        self
    }

    pub fn single_device(mut self) -> Self {
        self.planning_policy = PlanningPolicy::SingleDevice;
        self
    }

    pub fn auto_sharding(mut self, mesh: crate::Mesh, max_search_states: usize) -> Result<Self> {
        self.planning_policy =
            PlanningPolicy::Auto(crate::AutoShardingOptions::new(mesh, max_search_states)?);
        Ok(self)
    }

    pub fn build(self) -> Result<Runtime> {
        if self.backends.is_empty() {
            return Err(err("runtime builder requires a PJRT backend"));
        }
        let mut backends = BTreeMap::new();
        for registration in self.backends {
            let name = registration.name;
            if name.is_empty() {
                return Err(err("runtime backend name must be nonempty"));
            }
            let backend = backend_from_client(
                &name,
                registration.client,
                self.cache_limits,
                #[cfg(feature = "disk-cache")]
                registration.disk_cache,
            )?;
            if backends.insert(name.clone(), backend).is_some() {
                return Err(err(format!(
                    "runtime backend {name:?} is registered more than once"
                )));
            }
        }
        let default_backend = match self.default_backend {
            Some(name) => name,
            None if backends.len() == 1 => backends.keys().next().unwrap().clone(),
            None => return Err(err("multiple backends require a default backend")),
        };
        let default_device = backends
            .get(&default_backend)
            .ok_or_else(|| {
                err(format!(
                    "default runtime backend {default_backend:?} is not registered"
                ))
            })?
            .devices[backends[&default_backend].default_ordinal]
            .clone();
        Ok(Runtime {
            backends,
            default_device,
            planning_policy: self.planning_policy,
        })
    }
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Load one trusted PJRT plugin and construct a single-backend runtime.
    ///
    /// This is the concise path for ordinary CPU- or accelerator-only programs.
    /// Use [`Self::builder`] with explicit clients for heterogeneous runtimes.
    ///
    /// # Safety
    /// The resolved library must be trusted and implement the PJRT ABI.
    pub unsafe fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(unsafe { Client::load(path) }?))
    }

    /// Load one trusted PJRT plugin with plugin-specific client options.
    ///
    /// # Safety
    /// The resolved library and its handling of option data must satisfy
    /// [`Client::load_with_options`].
    pub unsafe fn load_with_options(
        path: impl AsRef<std::path::Path>,
        options: &ClientOptions,
    ) -> Result<Self> {
        Ok(Self::new(unsafe {
            Client::load_with_options(path, options)
        }?))
    }

    /// Load one trusted PJRT plugin and attach a trusted persistent executable
    /// cache to its default backend.
    ///
    /// # Safety
    /// The plugin must satisfy [`Self::load`]'s contract. The cache directory,
    /// compatibility key, and loaded client must additionally satisfy
    /// [`DiskCache::new_for_client`]'s contract.
    #[cfg(feature = "disk-cache")]
    pub unsafe fn load_with_cache(
        plugin: impl AsRef<std::path::Path>,
        cache_directory: impl AsRef<std::path::Path>,
        compatibility_key: &str,
        max_entry_bytes: usize,
    ) -> Result<Self> {
        let client = unsafe { Client::load(plugin) }?;
        unsafe {
            Self::builder().cached_client(
                client,
                cache_directory,
                compatibility_key,
                max_entry_bytes,
            )
        }?
        .build()
    }

    pub fn new(client: Client) -> Self {
        Self::with_cache_limits(client, CacheLimits::default())
    }

    pub fn with_cache_limits(client: Client, limits: CacheLimits) -> Self {
        Self::with_planning_policy(client, limits, PlanningPolicy::SingleDevice)
    }

    pub fn with_planning_policy(
        client: Client,
        limits: CacheLimits,
        planning_policy: PlanningPolicy,
    ) -> Self {
        let mut runtime = Self::builder()
            .client(client)
            .cache_limits(limits)
            .build()
            .expect("single-client runtime configuration is valid");
        runtime.planning_policy = planning_policy;
        runtime
    }

    pub fn planning_policy(&self) -> &PlanningPolicy {
        &self.planning_policy
    }

    pub fn plan(&self, program: &Program) -> Result<ExecutionPlan> {
        program.plan(self.planning_policy.clone())
    }

    /// Build a reusable explicit program without exposing Graph.
    pub fn trace<F>(&mut self, build: F) -> Result<Program>
    where
        F: FnOnce(&Tracer) -> Result<Vec<Tensor>>,
    {
        Tracer::trace(build)
    }

    pub fn client(&self) -> &Client {
        self.client_on(&self.default_device)
            .expect("the runtime default device is always registered")
    }

    fn client_on(&self, device: &Device) -> Result<&Client> {
        Ok(self.backend_for_device(device)?.compiler.client())
    }

    fn backend_for_device(&self, device: &Device) -> Result<&Backend> {
        let backend = self.backends.get(device.backend()).ok_or_else(|| {
            err(format!(
                "runtime backend {:?} is not registered",
                device.backend()
            ))
        })?;
        if backend.devices.get(device.ordinal()) != Some(device) {
            return Err(err(format!(
                "device {device:?} does not belong to this runtime backend"
            )));
        }
        Ok(backend)
    }

    pub fn backend_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.backends.keys().map(String::as_str)
    }

    pub fn devices(&self, backend: impl AsRef<str>) -> Result<&[Device]> {
        self.backends
            .get(backend.as_ref())
            .map(|backend| backend.devices.as_slice())
            .ok_or_else(|| {
                err(format!(
                    "runtime backend {:?} is not registered",
                    backend.as_ref()
                ))
            })
    }

    pub fn default_device(&self) -> &Device {
        &self.default_device
    }

    pub fn set_default_device(&mut self, device: &Device) -> Result<()> {
        self.backend_for_device(device)?;
        self.default_device = device.clone();
        Ok(())
    }

    pub fn on(&mut self, device: &Device) -> Result<DeviceRuntime<'_>> {
        self.client_on(device)?;
        Ok(DeviceRuntime {
            runtime: self,
            device: device.clone(),
        })
    }

    pub fn stats(&self) -> CacheStats {
        self.stats_on(&self.default_device)
            .expect("the runtime default device is always registered")
    }

    /// Release executable cache entries owned by every registered backend.
    /// Already returned executable handles remain valid.
    pub fn clear(&mut self) {
        for backend in self.backends.values_mut() {
            backend.compiler.clear();
        }
    }

    fn stats_on(&self, device: &Device) -> Result<CacheStats> {
        Ok(self.backend_for_device(device)?.compiler.stats())
    }

    pub fn compile(&mut self, program: &Program) -> Result<Arc<Executable>> {
        let device = self.default_device.clone();
        self.compile_on(&device, program)
    }

    pub(crate) fn compile_state_graph(
        &mut self,
        graph: &StateGraph,
        outputs: &[Tensor],
    ) -> Result<StateProgram> {
        let device = self.default_device.clone();
        let backend = self
            .backends
            .get_mut(device.backend())
            .ok_or_else(|| err(format!("runtime device {device:?} is not registered")))?;
        graph.compile(&mut backend.compiler, outputs)
    }

    fn compile_on(&mut self, device: &Device, program: &Program) -> Result<Arc<Executable>> {
        let plan = self.plan(program)?;
        let target_lowered = program.lower_for_compile(rxla_ir::LoweringTarget::Portable)?;
        if plan.requires_spmd_partitioning() {
            let use_shardy = target_lowered.is_stablehlo()
                && plan
                    .shardings()
                    .iter()
                    .any(|sharding| sharding.is_constraint());
            let config = auto_spmd_config(&plan, self.client_on(device)?, use_shardy)?;
            let lowered =
                program
                    .planning
                    .lower_spmd(&target_lowered, &plan, &config.device_ids)?;
            return self
                .backends
                .get_mut(device.backend())
                .expect("client_on validated the backend")
                .compiler
                .compile_lowered_with_options(&lowered, &config.options);
        }
        let options = single_device_options(device);
        self.backends
            .get_mut(device.backend())
            .ok_or_else(|| err(format!("runtime device {device:?} is not registered")))?
            .compiler
            .compile_lowered_with_options(&target_lowered, &options)
    }

    pub fn run(&mut self, program: &Program, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let device = self.default_device.clone();
        self.run_on(&device, program, inputs)
    }

    pub fn run_on(
        &mut self,
        device: &Device,
        program: &Program,
        inputs: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.len() != program.input_count() {
            return Err(err("input count mismatch"));
        }
        let client = self.client_on(device)?.clone();
        let buffers = inputs
            .iter()
            .enumerate()
            .map(|(index, values)| {
                let spec = program
                    .input_spec(index)
                    .expect("program input count matches input specs");
                if spec.dtype != DType::F32 {
                    return Err(err("non-F32 inputs require run_buffers"));
                }
                Ok(client.buffer_on_device(device.ordinal(), spec.shape, values)?)
            })
            .collect::<Result<Vec<_>>>()?;
        self.compile_on(device, program)?
            .execute_replicated(&buffers.iter().collect::<Vec<_>>())?
            .iter()
            .map(|buffer| Ok(buffer.to_vec::<f32>()?))
            .collect()
    }

    /// Execute typed resident inputs and return typed resident outputs.
    pub fn run_buffers(&mut self, program: &Program, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        let device = self.default_device.clone();
        self.run_buffers_on(&device, program, inputs)
    }

    pub fn run_buffers_on(
        &mut self,
        device: &Device,
        program: &Program,
        inputs: &[&Buffer],
    ) -> Result<Vec<Buffer>> {
        let client = self.client_on(device)?.clone();
        for (index, buffer) in inputs.iter().enumerate() {
            if !buffer.belongs_to(&client) || buffer.device_index()? != device.ordinal() {
                return Err(err(format!(
                    "input {index} is not resident on runtime device {device:?}"
                )));
            }
        }
        self.compile_on(device, program)?.execute_replicated(inputs)
    }

    pub fn run_tensors(&mut self, program: &Program, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let device = self.default_device.clone();
        self.run_tensors_on(&device, program, inputs)
    }

    pub fn run_tensors_on(
        &mut self,
        device: &Device,
        program: &Program,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != program.input_count() {
            return Err(err("input count mismatch"));
        }
        for (index, input) in inputs.iter().enumerate() {
            let expected = program
                .input_spec(index)
                .expect("program input count matches input specs");
            if input.shape() != expected.shape || input.dtype() != expected.dtype {
                return Err(err(format!("input {index}: tensor metadata mismatch")));
            }
        }
        let client = self.client_on(device)?.clone();
        let buffers = inputs
            .iter()
            .map(|input| input.to_buffer_on_device(&client, device.ordinal()))
            .collect::<Result<Vec<_>>>()?;
        self.run_buffers_on(
            device,
            program,
            &buffers
                .iter()
                .map(|buffer| buffer.as_ref())
                .collect::<Vec<_>>(),
        )?
        .into_iter()
        .map(Tensor::materialized)
        .collect()
    }

    /// Materialize one or more lazy expressions with a typed return shape.
    pub fn eval<E: Evaluable>(&mut self, values: E) -> Result<E::Output> {
        let device = self.default_device.clone();
        values.eval_with(self, &device)
    }

    fn eval_tensor_on(&mut self, device: &Device, output: &Tensor) -> Result<Tensor> {
        if output.is_materialized() {
            return Ok(output.clone());
        }
        Ok(self
            .eval_many_on(device, std::slice::from_ref(output))?
            .remove(0))
    }

    /// Materialize several outputs together, preserving shared computation.
    pub fn eval_many(&mut self, outputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let device = self.default_device.clone();
        self.eval_many_on(&device, outputs)
    }

    fn eval_many_on(&mut self, device: &Device, outputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        let pending = outputs
            .iter()
            .filter(|output| !output.is_materialized())
            .cloned()
            .collect::<Vec<_>>();
        let Some(first_pending) = pending.first() else {
            return Ok(outputs.to_vec());
        };
        let graph = first_pending.graph();
        if pending
            .iter()
            .any(|output| !std::sync::Arc::ptr_eq(&graph.0, &output.graph().0))
        {
            return Err(Error::LazySessionMismatch);
        }
        let roots = pending.clone();
        let (lowered, planning, parameters, source) = graph.direct_program(&roots, false)?;
        let inputs = first_pending.lazy_inputs(&parameters)?;
        let program = Program {
            planning,
            lowered,
            source,
        };
        let values = self.run_tensors_on(device, &program, &inputs.iter().collect::<Vec<_>>())?;
        Tensor::materialize_all(&pending, &values)?;
        Ok(outputs.to_vec())
    }
}

impl DeviceRuntime<'_> {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn client(&self) -> &Client {
        self.runtime
            .client_on(&self.device)
            .expect("a backend view retains a registered device")
    }

    pub fn stats(&self) -> CacheStats {
        self.runtime
            .stats_on(&self.device)
            .expect("a backend view retains a registered device")
    }

    /// Copy a resident buffer to this device through explicitly synchronized
    /// host staging. This also works across distinct PJRT implementations.
    pub fn transfer(&self, buffer: &Buffer) -> Result<Buffer> {
        Ok(buffer.copy_to_device_via_host(self.client(), self.device.ordinal())?)
    }

    pub fn transfer_with_limit(&self, buffer: &Buffer, max_bytes: usize) -> Result<Buffer> {
        Ok(buffer.copy_to_device_via_host_with_limit(
            self.client(),
            self.device.ordinal(),
            max_bytes,
        )?)
    }

    pub fn compile(&mut self, program: &Program) -> Result<Arc<Executable>> {
        self.runtime.compile_on(&self.device, program)
    }

    pub fn run(&mut self, program: &Program, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        self.runtime.run_on(&self.device, program, inputs)
    }

    pub fn run_buffers(&mut self, program: &Program, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        self.runtime.run_buffers_on(&self.device, program, inputs)
    }

    pub fn run_tensors(&mut self, program: &Program, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        self.runtime.run_tensors_on(&self.device, program, inputs)
    }

    pub fn eval<E: Evaluable>(&mut self, values: E) -> Result<E::Output> {
        values.eval_with(self.runtime, &self.device)
    }
}

fn backend_from_client(
    name: &str,
    client: Client,
    limits: CacheLimits,
    #[cfg(feature = "disk-cache")] disk_cache: Option<DiskCache>,
) -> Result<Backend> {
    let info = client.info()?;
    let default_ordinal = info
        .addressable_devices
        .iter()
        .position(|device| device.selected)
        .ok_or_else(|| err("PJRT client has no selected device"))?;
    let devices = info
        .addressable_devices
        .iter()
        .enumerate()
        .map(|(ordinal, device)| Device {
            backend: name.to_owned(),
            ordinal,
            id: device.id,
            kind: device.kind.clone(),
        })
        .collect::<Vec<_>>();
    if devices.is_empty() {
        return Err(err("PJRT backend has no addressable devices"));
    }
    let compiler = Compiler::new(client, limits);
    #[cfg(feature = "disk-cache")]
    let compiler = match disk_cache {
        Some(cache) => compiler.with_disk_cache(cache),
        None => compiler,
    };
    Ok(Backend {
        compiler,
        devices,
        default_ordinal,
    })
}

fn single_device_options(device: &Device) -> Vec<u8> {
    CompileOptionsProto {
        executable_build_options: Some(ExecutableBuildOptionsProto {
            num_replicas: 1,
            num_partitions: 1,
            device_ordinal: -1,
            device_assignment: Some(DeviceAssignmentProto {
                replica_count: 1,
                computation_count: 1,
                computation_devices: vec![ComputationDevice {
                    replica_device_ids: vec![i64::from(device.id())],
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

struct AutoSpmdConfig {
    options: Vec<u8>,
    device_ids: Vec<i64>,
}

fn auto_spmd_config(
    plan: &ExecutionPlan,
    client: &Client,
    use_shardy: bool,
) -> Result<AutoSpmdConfig> {
    let crate::PlanningPolicy::Auto(options) = plan.policy() else {
        return Err(err("multi-device plan requires auto-sharding policy"));
    };
    let required = plan.required_devices();
    let info = client.info()?;
    if info.addressable_devices.len() < required {
        return Err(err(format!(
            "auto-sharding mesh requires {required} devices, but client has {}",
            info.addressable_devices.len()
        )));
    }
    let device_ids = info
        .addressable_devices
        .iter()
        .take(required)
        .map(|device| i64::from(device.id))
        .collect::<Vec<_>>();
    let build = ExecutableBuildOptionsProto {
        num_replicas: 1,
        num_partitions: required as i64,
        device_ordinal: -1,
        device_assignment: Some(DeviceAssignmentProto {
            replica_count: 1,
            computation_count: required as i32,
            computation_devices: device_ids
                .iter()
                .map(|&device| ComputationDevice {
                    replica_device_ids: vec![device],
                })
                .collect(),
        }),
        use_spmd_partitioning: true,
        use_auto_spmd_partitioning: !use_shardy,
        use_shardy_partitioner: use_shardy,
        optimization_level: options.effort_level() as i32,
        auto_spmd_partitioning_mesh_shape: options
            .mesh()
            .axes()
            .iter()
            .map(|axis| axis.size() as i64)
            .collect(),
        auto_spmd_partitioning_mesh_ids: device_ids.clone(),
        ..Default::default()
    };
    let options = CompileOptionsProto {
        executable_build_options: Some(build),
        ..Default::default()
    }
    .encode_to_vec();
    Ok(AutoSpmdConfig {
        options,
        device_ids,
    })
}

impl Tensor {
    /// Snapshot this lazy expression into verified StableHLO without a runtime.
    /// The returned program can be inspected, cached, compiled, and repeatedly
    /// called with compatible Tensor inputs.
    pub fn program(&self) -> Result<Program> {
        Program::from_tensors(std::slice::from_ref(self))
    }

    /// Evaluate this lazy expression and return its materialized value.
    pub fn eval(&self, runtime: &mut Runtime) -> Result<Self> {
        runtime.eval(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_can_be_owned_by_a_worker_thread() {
        fn assert_send<T: Send>() {}
        assert_send::<Runtime>();
    }

    #[test]
    fn tracer_builds_common_programs_directly_in_pliron() {
        let tracer = Tracer::new();
        let x = tracer.input(&[2]).unwrap();
        let bias = tracer.constant(&[2], &[1., 2.]).unwrap();
        let output = x.add(&bias).unwrap();
        let program = tracer.program(vec![output]).unwrap();
        assert_eq!(program.lowered.format(), "mlir");

        let tracer = Tracer::new();
        let lhs = tracer.input(&[2, 2]).unwrap();
        let rhs = tracer.input(&[2, 2]).unwrap();
        let matmul = lhs.matmul(&rhs).unwrap();
        let transposed = matmul.transpose(&[1, 0]).unwrap();
        let reversed = transposed.flip(&[0]).unwrap();
        let cumulative = reversed.cumsum(0).unwrap();
        let slice = cumulative.narrow(0, 0, 1).unwrap();
        let padded = slice.pad(&[[1, 1], [0, 0]], 0.).unwrap();
        let concatenated = Tensor::concatenate(&[padded.clone(), padded], 0).unwrap();
        let finite = concatenated.is_finite_mask().unwrap();
        let zero = tracer.constant(&[], &[0.]).unwrap();
        let zero = zero.broadcast_to(concatenated.shape()).unwrap();
        let positive = concatenated.gt_mask(&zero).unwrap();
        let selected = finite.select(&positive, &zero).unwrap();
        let maximum = selected.argmax(0, false).unwrap();
        assert_eq!(
            tracer.program(vec![maximum]).unwrap().lowered.format(),
            "mlir"
        );
    }

    #[test]
    fn tracer_rejects_unmigrated_training_ops_without_falling_back() {
        let tracer = Tracer::new();
        let input = tracer.input(&[1, 2, 2, 1]).unwrap();
        let kernel = tracer.input(&[1, 1, 1, 1]).unwrap();
        let error = tracer
            .graph
            .push_node(
                crate::Op::Conv2dInputGradient(Default::default()),
                vec![input.node_id(), kernel.node_id()],
                crate::TensorType {
                    dims: vec![1, 2, 2, 1],
                    dtype: crate::DType::F32,
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::Ir {
                source: crate::IrError::UnsupportedOperation { .. }
            }
        ));
        let graph = tracer.graph.0.lock().unwrap();
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn common_elementwise_trace_stays_native_pliron() {
        let tracer = Tracer::new();
        let x = tracer.input(&[2]).unwrap();
        let y = tracer.input(&[2]).unwrap();
        let output = x
            .exp()
            .unwrap()
            .add(&y)
            .unwrap()
            .tanh()
            .unwrap()
            .reshape(&[1, 2])
            .unwrap();
        tracer.program(vec![output]).unwrap();
    }

    #[test]
    fn attention_forward_stays_native_pliron() {
        let tracer = Tracer::new();
        let query = tracer.input(&[2, 3, 4]).unwrap();
        let key = tracer.input(&[2, 5, 4]).unwrap();
        let value = tracer.input(&[2, 5, 6]).unwrap();
        let output = query
            .scaled_dot_product_attention(&key, &value, None, None)
            .unwrap();
        assert_eq!(output.shape(), [2, 3, 6]);
        let nodes = tracer.graph.0.lock().unwrap().semantic_nodes().unwrap();
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node.op, rxla_ir::Op::Attention { .. }))
        );
        let program = tracer.program(vec![output]).unwrap();
        assert_eq!(program.lowered.format(), "mlir");
        let mlir = str::from_utf8(program.lowered.code()).unwrap();
        assert_eq!(mlir.matches("stablehlo.dot_general").count(), 2);
        assert!(mlir.contains("stablehlo.reduce"));
    }

    #[test]
    fn convolution_forward_stays_native_pliron() {
        let tracer = Tracer::new();
        let input = tracer.input(&[1, 5, 6, 4]).unwrap();
        let kernel = tracer.input(&[2, 3, 2, 6]).unwrap();
        let output = input
            .conv2d(
                &kernel,
                crate::Conv2dOptions {
                    strides: [2, 1],
                    padding: [[1, 0], [0, 1]],
                    dilation: [1, 2],
                    groups: 2,
                },
            )
            .unwrap();
        assert_eq!(output.shape(), [1, 3, 3, 6]);
        let program = tracer.program(vec![output]).unwrap();
        assert_eq!(program.lowered.format(), "mlir");
        let mlir = str::from_utf8(program.lowered.code()).unwrap();
        assert!(mlir.contains("stablehlo.convolution"));
    }

    #[test]
    fn unet_style_residual_block_stays_native_pliron() {
        let tracer = Tracer::new();
        let input = tracer.input(&[1, 4, 4, 4]).unwrap();
        let first_kernel = tracer.input(&[3, 3, 4, 4]).unwrap();
        let second_kernel = tracer.input(&[3, 3, 4, 4]).unwrap();
        let convolution = crate::Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            ..Default::default()
        };
        let hidden = input
            .conv2d(&first_kernel, convolution)
            .unwrap()
            .transpose(&[0, 3, 1, 2])
            .unwrap()
            .group_norm(2, None, None, 1e-5)
            .unwrap()
            .transpose(&[0, 2, 3, 1])
            .unwrap()
            .silu()
            .unwrap()
            .conv2d(&second_kernel, convolution)
            .unwrap();
        let output = hidden.add(&input).unwrap();

        assert_eq!(output.shape(), [1, 4, 4, 4]);
        tracer.program(vec![output]).unwrap();
    }
    use std::sync::Arc;

    #[test]
    fn tracing_facade_hides_graph_from_program_users() {
        let program = Tracer::trace(|trace| {
            let x = trace.input(&[2])?;
            let one = trace.constant(&[2], &[1.0, 1.0])?;
            Ok(vec![x.add(&one)?])
        })
        .unwrap();

        assert_eq!(program.input_count(), 1);
        assert_eq!(program.output_count(), 1);
        assert_eq!(program.output_spec(0).unwrap().shape, &[2]);
        assert_eq!(program.output_spec(0).unwrap().dtype, DType::F32);
    }

    #[test]
    fn runtime_builder_validates_configuration_before_backend_use() {
        assert!(Runtime::builder().build().is_err());
        let mesh = crate::Mesh::new([("data", 1)]).unwrap();
        assert!(Runtime::builder().auto_sharding(mesh, 0).is_err());
    }

    #[test]
    fn program_is_a_snapshot_and_does_not_retain_the_tracing_graph() {
        let tracer = Tracer::new();
        let weak = Arc::downgrade(&tracer.graph.0);
        let x = tracer.input(&[2]).unwrap();
        let program = tracer.program(vec![x]).unwrap();
        drop(tracer);
        assert!(weak.upgrade().is_none());
        assert_eq!(program.input_count(), 1);
    }

    #[test]
    fn native_program_prunes_and_renumbers_unreachable_pliron_values() {
        let tracer = Tracer::new();
        let mesh = crate::Mesh::new([("data", 2)]).unwrap();
        let _unused = tracer
            .input(&[4])
            .unwrap()
            .with_sharding(crate::Sharding::partitioned(
                mesh,
                crate::PartitionSpec::new([Some("data")]).unwrap(),
            ))
            .unwrap();
        let used = tracer.input(&[2]).unwrap();
        let program = tracer.program(vec![used.exp().unwrap()]).unwrap();

        assert_eq!(program.input_count(), 1);
        assert_eq!(program.input_spec(0).unwrap().shape, &[2]);
        assert!(program.plan(PlanningPolicy::SingleDevice).is_ok());
        assert!(program.planning.sharding_constraints().is_empty());
    }

    #[test]
    fn program_retains_pre_lowering_sharding_constraints() {
        let tracer = Tracer::new();
        let mesh = crate::Mesh::new([("data", 2)]).unwrap();
        let x = tracer
            .input(&[4, 2])
            .unwrap()
            .with_sharding(crate::Sharding::partitioned(
                mesh.clone(),
                crate::PartitionSpec::new([Some("data"), None]).unwrap(),
            ))
            .unwrap();
        let y = x
            .exp()
            .unwrap()
            .with_sharding(crate::Sharding::replicated(mesh))
            .unwrap();
        let program = tracer.program(vec![y]).unwrap();

        assert_eq!(program.input_count(), 1);
        assert_eq!(program.output_count(), 1);
        assert_eq!(program.planning.sharding_constraints().len(), 2);
    }
}
