//! Rust tensor programs represented in Pliron and compiled through PJRT.
pub use batch_norm_state::{BatchNormState, BatchNormUpdate};
pub use compiler::{CacheLimits, CacheStats, Compiler, LoweredProgram};
pub use dlpark;
pub use half::{bf16, f16};
pub use kv_cache::{KvCache, KvCacheUpdate};
pub use nn::BatchNormTraining;
use prost::Message;
pub use rxla_pjrt::{
    Buffer, BufferMemoryLayout, Client, ClientInfo, ClientOptionValue, ClientOptions, DType,
    DeviceInfo, Element, Error as PjrtError, MemoryInfo, PendingExecution, PendingHostUpload,
    PjrtErrorCode, Plugin, PluginRegistry,
};
use rxla_xla_proto::xla::{
    CompileOptionsProto, DeviceAssignmentProto, ExecutableBuildOptionsProto, HloModuleProto,
    PrimitiveType,
};
use snafu::Snafu;
pub use state::{
    Parameter, ParameterId, PreparedStateGraph, Session, StateGraph, StateProgram, StateSlot,
};
use std::sync::{Arc, Mutex};
pub use typed_state::{F32, I32, State, StateDType, StateTransaction, StateUpdates};
mod artifact;
mod attention;
mod autodiff;
mod batch_norm_state;
mod boxes;
mod compiler;
mod contraction;
mod convolution;
#[cfg(feature = "disk-cache")]
mod disk_cache;
mod frontend;
mod image;
mod shape_ops;
#[cfg(feature = "disk-cache")]
pub use disk_cache::{DiskCache, DiskCacheInspection, DiskCacheTrim};
mod indexing;
mod kv_cache;
mod nn;
mod operators;
mod ops;
mod planning;
mod pooling;
pub mod random;
mod resize;
mod rotary;
pub mod vision;
pub use planning::{
    AutoShardingOptions, ExecutionPlan, ExecutionStage, PlannedSharding, PlanningPolicy,
    ShardingConstraint, ShardingDecision,
};
pub use rotary::RotaryLayout;
pub use rxla_ir::{Mesh, MeshAxis, PartitionSpec, Sharding, ShardingError};
mod state;
pub mod state_tree;
mod transposed_convolution;
mod typed_state;
pub use rxla_ir::IrError;
use rxla_ir::{Binary, Op, Reduction, SliceAxis, TensorType, Unary};
pub use rxla_ir::{Conv2dOptions, ConvTranspose2dOptions, Pool2dOptions};

/// One tensor axis in a traced signature. Rank and upper bounds remain compile-
/// time information while a bounded axis obtains its actual size at execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Static(i64),
    Bounded { upper: i64 },
}
/// Public tensor-layer failures. Subsystems own their concrete error types;
/// this boundary only composes them for operations spanning multiple layers.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(transparent)]
    Pjrt { source: PjrtError },
    #[snafu(transparent)]
    Ir { source: IrError },
    #[snafu(transparent)]
    Sharding { source: ShardingError },
    #[snafu(transparent)]
    Storage { source: StorageError },
    #[snafu(display("tensor has no expression node"))]
    MissingExpression,
    #[snafu(display("tensor belongs to an explicit Tracer; execute its Program"))]
    ExplicitTraceEvaluation,
    #[snafu(display("lazy input binding {index} is missing"))]
    MissingLazyInput { index: usize },
    #[snafu(display("executor returned {actual} materialized outputs, expected {expected}"))]
    MaterializedOutputCount { expected: usize, actual: usize },
    #[snafu(display("materialized output {index} does not match requested shape or dtype"))]
    MaterializedOutputMetadata { index: usize },
    #[snafu(display("executor output {index} is not materialized"))]
    ExecutorOutputNotMaterialized { index: usize },
    #[snafu(display("tensors belong to different implicit lazy sessions"))]
    LazySessionMismatch,
    #[snafu(display("lazy output {index} already has an evaluation in flight"))]
    EvaluationInFlight { index: usize },
    #[snafu(display("an evaluation lease requires at least one lazy output"))]
    EmptyEvaluationLease,
    #[snafu(display("a reusable Program requires at least one Tensor output"))]
    ProgramOutputRequired,
    #[snafu(display("program output {index} belongs to a different lazy trace"))]
    ProgramOutputTraceMismatch { index: usize },
    #[snafu(display("evaluation returned {actual} outputs, expected {expected}"))]
    EvaluationOutputCount { expected: usize, actual: usize },
    #[snafu(display(
        "asynchronous Tensor evaluation currently requires one executable device, got {actual}"
    ))]
    AsyncEvaluationDeviceCount { actual: usize },
    #[snafu(display("tensor graph lock is poisoned"))]
    GraphLockPoisoned,
    #[snafu(display("invalid bounded tensor shape: {message}"))]
    InvalidBoundedShape { message: String },
    #[snafu(display("storage can only be bound to a symbolic input"))]
    StorageBindingRequiresInput,
    #[snafu(display("declared tensor dtype {declared:?} does not match storage dtype {actual:?}"))]
    StorageDTypeMismatch { declared: DType, actual: DType },
    #[snafu(display(
        "host input expects {expected_shape:?} {expected_dtype:?}, received {actual_shape:?} {actual_dtype:?}"
    ))]
    HostInputMetadata {
        expected_shape: Vec<i64>,
        expected_dtype: DType,
        actual_shape: Vec<i64>,
        actual_dtype: DType,
    },
    #[snafu(display("operation requires PJRT device storage"))]
    ExpectedDeviceStorage,
    #[snafu(display(
        "device input expects {expected_shape:?} {expected_dtype:?}, received {actual_shape:?} {actual_dtype:?}"
    ))]
    DeviceInputMetadata {
        expected_shape: Vec<i64>,
        expected_dtype: DType,
        actual_shape: Vec<i64>,
        actual_dtype: DType,
    },
    #[snafu(display("tensor has no host storage"))]
    MissingHostStorage,
    #[snafu(display("managed tensor belongs to a different PJRT client"))]
    ForeignClientStorage,
    #[snafu(display("symbolic tensor has no managed storage"))]
    MissingManagedStorage,
    #[snafu(display("single-output execution requires 1 output, executable has {actual}"))]
    SingleOutputRequired { actual: usize },
    #[snafu(display("F32 convenience execution does not support input {index} dtype {dtype:?}"))]
    F32InputRequired { index: usize, dtype: DType },
    #[snafu(display("executable expects {expected} inputs, received {actual}"))]
    ExecutableInputCount { expected: usize, actual: usize },
    #[snafu(display("executable input {index} expects shape {expected:?}, received {actual:?}"))]
    ExecutableInputShape {
        index: usize,
        expected: Vec<i64>,
        actual: Vec<i64>,
    },
    #[snafu(display("executable input {index} expects dtype {expected:?}, received {actual:?}"))]
    ExecutableInputDType {
        index: usize,
        expected: DType,
        actual: DType,
    },
    #[snafu(display("program is missing metadata for input {index}"))]
    MissingProgramInputSpec { index: usize },
    #[snafu(display(
        "runtime input {index} belongs to another PJRT client, expected backend {backend:?}"
    ))]
    RuntimeInputClient { index: usize, backend: String },
    #[snafu(display(
        "runtime input {index} is on device ordinal {actual}, expected {expected} for backend {backend:?}"
    ))]
    RuntimeInputDevice {
        index: usize,
        backend: String,
        expected: usize,
        actual: usize,
    },
    #[snafu(display(
        "sharded execution device {device} returned {actual} outputs, expected {expected}"
    ))]
    ShardedOutputCount {
        device: usize,
        expected: usize,
        actual: usize,
    },
    #[snafu(display("invalid optimized program {format:?} protobuf: {source}"))]
    OptimizedProgramDecode {
        format: String,
        source: prost::DecodeError,
    },
    #[snafu(display("optimized program with configuration has no HLO module"))]
    MissingOptimizedHloModule,
    #[snafu(display("unsupported optimized program format {format:?}"))]
    UnsupportedOptimizedProgram { format: String },
    #[snafu(display("runtime requires at least one PJRT backend"))]
    MissingRuntimeBackend,
    #[snafu(display("runtime backend name must be nonempty"))]
    EmptyRuntimeBackendName,
    #[snafu(display("runtime backend {name:?} is registered more than once"))]
    DuplicateRuntimeBackend { name: String },
    #[snafu(display("multiple runtime backends require an explicit default backend"))]
    MissingDefaultRuntimeBackend,
    #[snafu(display("runtime backend {name:?} is not registered"))]
    RuntimeBackendNotRegistered { name: String },
    #[snafu(display(
        "device ordinal {ordinal} (id {id}, kind {kind:?}) is not registered on runtime backend {backend:?}"
    ))]
    RuntimeDeviceNotRegistered {
        backend: String,
        ordinal: usize,
        id: i32,
        kind: String,
    },
    #[snafu(display("PJRT backend {backend:?} has no selected device"))]
    RuntimeBackendWithoutSelectedDevice { backend: String },
    #[snafu(display("PJRT backend {backend:?} has no addressable devices"))]
    RuntimeBackendWithoutDevices { backend: String },
    #[snafu(display("stateful model step has no compiled program"))]
    MissingStatefulProgram,
    #[snafu(display("compiled execution argument refers to pruned input {index}"))]
    PrunedExecutionInput { index: usize },
    #[snafu(display("invalid tensor operation: {message}"))]
    InvalidArgument { message: String },
    #[snafu(display("{operation} axis {axis} is out of range for a rank-{rank} tensor"))]
    AxisOutOfRange {
        operation: &'static str,
        axis: usize,
        rank: usize,
    },
    #[snafu(display("conditional predicate must be a scalar I32 tensor"))]
    InvalidConditionalPredicate,
    #[snafu(display("conditional branches must belong to the predicate trace"))]
    ConditionalTraceMismatch,
    #[snafu(display(
        "conditional branch result counts differ: then has {then_count}, else has {else_count}"
    ))]
    ConditionalResultCount {
        then_count: usize,
        else_count: usize,
    },
    #[snafu(display(
        "conditional branch result {index} differs: then is {then_shape:?} {then_dtype:?}, else is {else_shape:?} {else_dtype:?}"
    ))]
    ConditionalResultMismatch {
        index: usize,
        then_shape: Vec<i64>,
        then_dtype: DType,
        else_shape: Vec<i64>,
        else_dtype: DType,
    },
    #[snafu(display("custom call target must not be empty"))]
    EmptyCustomCallTarget,
    #[snafu(display("a tensor custom call requires at least one tensor operand"))]
    EmptyCustomCallOperands,
    #[snafu(display("all custom call operands must belong to one trace"))]
    CustomCallTraceMismatch,
}

/// Result type shared by tensor construction, transformation and execution APIs.
pub type Result<T> = std::result::Result<T, Error>;
pub mod dlpack;
mod dtype_rules;
mod metadata;
mod tensor_handle;
pub use frontend::{
    Device, DeviceRuntime, Evaluable, PendingEvaluation, PendingTensorEvaluation, Program, Runtime,
    RuntimeBuilder, TensorFunction, Tracer,
};
pub use tensor_handle::{
    CustomCall, Storage, StorageError, StorageKind, StorageResult, TensorBuildError, TensorBuilder,
    TensorDescriptor, TensorDownloadError, TensorElement, TensorLayout,
};
fn err(message: impl Into<String>) -> Error {
    Error::InvalidArgument {
        message: message.into(),
    }
}

struct GraphInner {
    ir: Mutex<rxla_ir::ProgramIr>,
    lazy: std::sync::OnceLock<std::sync::Weak<tensor_handle::LazySession>>,
}

impl std::ops::Deref for GraphInner {
    type Target = Mutex<rxla_ir::ProgramIr>;

    fn deref(&self) -> &Self::Target {
        &self.ir
    }
}

#[derive(Clone)]
pub(crate) struct Graph(Arc<GraphInner>);

impl Default for Graph {
    fn default() -> Self {
        Self(Arc::new(GraphInner {
            ir: Mutex::new(rxla_ir::ProgramIr::default()),
            lazy: std::sync::OnceLock::new(),
        }))
    }
}
/// One-pointer, immutable, thread-safe tensor descriptor with optional managed
/// input storage. Clones share the descriptor; attaching storage creates a new
/// descriptor. Operators remain lazy and perform no automatic upload.
#[derive(Clone)]
pub struct Tensor {
    descriptor: Arc<TensorDescriptor>,
}

fn elements(dims: &[i64]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |n, &d| {
            usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
        })
        .ok_or_else(|| err("invalid or overflowing shape"))
}

fn validate_tensor_type(ty: &TensorType) -> Result<()> {
    if ty.dynamic_bounds.is_empty() {
        elements(&ty.dims)?;
        return Ok(());
    }
    if ty.dynamic_bounds.len() != ty.dims.len() {
        return Err(Error::InvalidBoundedShape {
            message: "bounds must have the same rank as the shape".into(),
        });
    }
    for (axis, (&dim, &bound)) in ty.dims.iter().zip(&ty.dynamic_bounds).enumerate() {
        let valid = (dim >= 0 && bound == -1) || (dim == -1 && bound > 0);
        if !valid {
            return Err(Error::InvalidBoundedShape {
                message: format!("axis {axis} has dimension {dim} and bound {bound}"),
            });
        }
    }
    Ok(())
}
impl Graph {
    fn region_marker(&self) -> Result<usize> {
        Ok(self
            .0
            .lock()
            .map_err(|_| Error::GraphLockPoisoned)?
            .region_marker())
    }

    fn conditional(
        &self,
        predicate: rxla_ir::SsaId,
        then_marker: usize,
        then_values: &[rxla_ir::SsaId],
        else_marker: usize,
        else_values: &[rxla_ir::SsaId],
        types: &[TensorType],
    ) -> Result<Vec<Tensor>> {
        let ids = self
            .0
            .lock()
            .map_err(|_| Error::GraphLockPoisoned)?
            .append_conditional(
                predicate,
                then_marker,
                then_values,
                else_marker,
                else_values,
                types,
            )?;
        Ok(ids
            .into_iter()
            .zip(types)
            .map(|(id, ty)| Tensor::symbolic_typed(self.clone(), id, ty))
            .collect())
    }

    fn set_sharding(&self, id: rxla_ir::SsaId, sharding: Sharding) -> Result<()> {
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        if graph.value_type(id).is_err() {
            return Err(err("invalid tensor expression node"));
        }
        Ok(graph.set_sharding(id, &sharding)?)
    }

    fn sharding(&self, id: rxla_ir::SsaId) -> Result<Option<Sharding>> {
        let graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        Ok(graph.sharding(id)?)
    }
    fn push_node(
        &self,
        op: Op,
        operands: Vec<rxla_ir::SsaId>,
        ty: TensorType,
    ) -> Result<rxla_ir::SsaId> {
        validate_tensor_type(&ty)?;
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let operand_types = graph
            .operand_types(&operands)
            .map_err(|_| err("operation requires tensors from an active tracing session"))?;
        let dtype = dtype_rules::infer(&op, &operand_types)?;
        if dtype != ty.dtype {
            return Err(err("operation output dtype mismatch"));
        }
        Ok(graph.append(&op, &operands, &ty)?)
    }
    fn node(&self, op: Op, operands: Vec<rxla_ir::SsaId>, dims: &[i64]) -> Result<Tensor> {
        let (dtype, dynamic_bounds) = {
            let graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
            let dtype = dtype_rules::infer(&op, &graph.operand_types(&operands)?)?;
            let bounds = operands
                .first()
                .and_then(|&operand| graph.value_type(operand).ok())
                .filter(|ty| ty.dims == dims)
                .map_or_else(Vec::new, |ty| ty.dynamic_bounds);
            (dtype, bounds)
        };
        let ty = TensorType {
            dims: dims.to_vec(),
            dtype,
            dynamic_bounds,
        };
        self.node_typed(op, operands, ty)
    }

    fn node_typed(&self, op: Op, operands: Vec<rxla_ir::SsaId>, ty: TensorType) -> Result<Tensor> {
        let id = self.push_node(op, operands, ty.clone())?;
        Ok(Tensor::symbolic_typed(self.clone(), id, &ty))
    }
    pub fn input(&self, dims: &[i64]) -> Result<Tensor> {
        self.input_dtype(dims, DType::F32)
    }
    /// Declare an input with an explicit runtime element type.
    pub fn input_dtype(&self, dims: &[i64], dtype: DType) -> Result<Tensor> {
        let id = self.parameter(TensorType {
            dims: dims.to_vec(),
            dtype,
            dynamic_bounds: vec![],
        })?;
        Ok(Tensor::symbolic(self.clone(), id, dims, dtype))
    }
    /// Declare an input whose dynamic axes have explicit compile-time upper bounds.
    pub fn input_shape(&self, dims: &[Dim], dtype: DType) -> Result<Tensor> {
        let mut shape = Vec::with_capacity(dims.len());
        let mut bounds = Vec::with_capacity(dims.len());
        let mut dynamic = false;
        for dim in dims {
            match *dim {
                Dim::Static(value) => {
                    shape.push(value);
                    bounds.push(-1);
                }
                Dim::Bounded { upper } => {
                    shape.push(-1);
                    bounds.push(upper);
                    dynamic = true;
                }
            }
        }
        if !dynamic {
            bounds.clear();
        }
        let ty = TensorType::bounded(shape, dtype, bounds);
        let id = self.parameter(ty.clone())?;
        Ok(Tensor::symbolic_typed(self.clone(), id, &ty))
    }
    /// Declare BF16 input storage and explicitly convert it to an F32 Tensor
    /// inside the compiled graph. Useful for frozen BF16 weights. Finite BF16
    /// values widen exactly; nonfinite handling is backend-defined. No host F32
    /// expansion is required. Backend scheduling/materialization is not promised.
    ///
    /// The returned conversion is not a trainable input leaf: `grad` with
    /// respect to it is rejected. Other F32 input/parameter gradients can depend
    /// on this value, but differentiation stops at the BF16 storage boundary.
    /// This is not mixed-precision arithmetic, quantization-aware training, or
    /// an implicit straight-through derivative for a BF16 master parameter.
    pub fn input_bf16_as_f32(&self, dims: &[i64]) -> Result<Tensor> {
        let id = self.parameter(TensorType {
            dims: dims.to_vec(),
            dtype: DType::BF16,
            dynamic_bounds: vec![],
        })?;
        self.node(Op::Bf16ToFloat, vec![id], dims)
    }
    fn parameter(&self, ty: TensorType) -> Result<rxla_ir::SsaId> {
        validate_tensor_type(&ty)?;
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let number = graph.parameter_count()?;
        let op = Op::Parameter(number);
        Ok(graph.append(&op, &[], &ty)?)
    }

    fn state_input(
        &self,
        dims: &[i64],
        dtype: DType,
        state_id: usize,
        path: &str,
    ) -> Result<Tensor> {
        elements(dims)?;
        let ty = TensorType {
            dims: dims.to_vec(),
            dtype,
            dynamic_bounds: vec![],
        };
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let number = graph.parameter_count()?;
        let id = graph.state_input(number, state_id, path, &ty)?;
        Ok(Tensor::symbolic(self.clone(), id, dims, dtype))
    }

    fn state_read(&self, current: &Tensor, state_id: usize) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.0, &current.graph().0) {
            return Err(err("state read belongs to another graph"));
        }
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let id = graph.state_read(current.node_id(), state_id)?;
        Ok(Tensor::symbolic(
            self.clone(),
            id,
            current.shape(),
            current.dtype(),
        ))
    }

    fn state_write(&self, value: &Tensor, state_id: usize) -> Result<Tensor> {
        if !Arc::ptr_eq(&self.0, &value.graph().0) {
            return Err(err("state write belongs to another graph"));
        }
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let id = graph.state_write(value.node_id(), state_id)?;
        Ok(Tensor::symbolic(
            self.clone(),
            id,
            value.shape(),
            value.dtype(),
        ))
    }

    fn direct_program(
        &self,
        outputs: &[Tensor],
        preserve_all_inputs: bool,
    ) -> Result<(
        LoweredProgram,
        planning::PlanningSnapshot,
        Vec<usize>,
        rxla_ir::SemanticProgram,
    )> {
        if outputs
            .iter()
            .any(|output| !Arc::ptr_eq(&self.0, &output.graph().0))
        {
            return Err(err("output belongs to another graph"));
        }
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let ir = &mut *graph;
        let output_ids = outputs.iter().map(Tensor::node_id).collect::<Vec<_>>();
        let planning = ir.planning_snapshot(&output_ids)?.into();
        let source = rxla_ir::SemanticProgram::capture(ir, &output_ids, preserve_all_inputs)?;
        let program = ir.stablehlo_program(&output_ids, preserve_all_inputs)?;
        let parameters = program.parameters.clone();
        let lowered = LoweredProgram::from_stablehlo(program.code, program.inputs, program.outputs);
        Ok((lowered, planning, parameters, source))
    }

    fn direct_lowered(
        &self,
        outputs: &[Tensor],
        preserve_all_inputs: bool,
    ) -> Result<LoweredProgram> {
        self.direct_lowered_for(
            outputs,
            preserve_all_inputs,
            rxla_ir::LoweringTarget::Portable,
        )
    }

    fn direct_lowered_for(
        &self,
        outputs: &[Tensor],
        preserve_all_inputs: bool,
        target: rxla_ir::LoweringTarget,
    ) -> Result<LoweredProgram> {
        if outputs
            .iter()
            .any(|output| !Arc::ptr_eq(&self.0, &output.graph().0))
        {
            return Err(err("output belongs to another graph"));
        }
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let ir = &mut *graph;
        let output_ids = outputs.iter().map(Tensor::node_id).collect::<Vec<_>>();
        let program = ir.stablehlo_program_for(&output_ids, preserve_all_inputs, target)?;
        Ok(LoweredProgram::from_stablehlo(
            program.code,
            program.inputs,
            program.outputs,
        ))
    }

    fn direct_lowered_pruned(&self, outputs: &[Tensor]) -> Result<(LoweredProgram, Vec<usize>)> {
        if outputs
            .iter()
            .any(|output| !Arc::ptr_eq(&self.0, &output.graph().0))
        {
            return Err(err("output belongs to another graph"));
        }
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        let ir = &mut *graph;
        let output_ids = outputs.iter().map(Tensor::node_id).collect::<Vec<_>>();
        let program = ir.stablehlo_program(&output_ids, false)?;
        let lowered = LoweredProgram::from_stablehlo(program.code, program.inputs, program.outputs);
        Ok((lowered, program.parameters))
    }
    pub fn constant(&self, dims: &[i64], values: &[f32]) -> Result<Tensor> {
        if elements(dims)? != values.len() {
            return Err(err("constant shape/data mismatch"));
        }
        self.node(Op::ConstantF32(values.into()), vec![], dims)
    }
    /// Export the Pliron representation as textual StableHLO MLIR.
    pub fn stablehlo(&self, output: &Tensor) -> Result<String> {
        self.stablehlo_many(std::slice::from_ref(output))
    }
    /// Export multiple F32 tensor results as textual StableHLO MLIR.
    pub fn stablehlo_many(&self, outputs: &[Tensor]) -> Result<String> {
        if outputs.is_empty() {
            return Err(err("at least one output is required"));
        }
        if outputs
            .iter()
            .any(|output| !Arc::ptr_eq(&self.0, &output.graph().0))
        {
            return Err(err("output belongs to another graph"));
        }
        let mut graph = self.0.lock().map_err(|_| Error::GraphLockPoisoned)?;
        Ok(graph.stablehlo(&outputs.iter().map(Tensor::node_id).collect::<Vec<_>>())?)
    }
    pub fn compile(&self, client: &Client, output: &Tensor) -> Result<Executable> {
        self.compile_many(client, std::slice::from_ref(output))
    }

    pub fn compile_many(&self, client: &Client, outputs: &[Tensor]) -> Result<Executable> {
        let lowered = self.prepare_many(outputs)?;
        lowered.compile_uncached(client)
    }
}

fn compile_program_with_options(
    client: &Client,
    format: &str,
    code: &[u8],
    inputs: &[TensorType],
    output_count: usize,
    compile_options: Option<&[u8]>,
) -> Result<Executable> {
    let selected_device = client
        .info()?
        .addressable_devices
        .into_iter()
        .find(|device| device.selected)
        .ok_or_else(|| err("client has no selected compilation device"))?;
    let default_options = CompileOptionsProto {
        executable_build_options: Some(ExecutableBuildOptionsProto {
            num_replicas: 1,
            num_partitions: 1,
            device_ordinal: -1,
            device_assignment: Some(DeviceAssignmentProto {
                replica_count: 1,
                computation_count: 1,
                computation_devices: vec![
                    rxla_xla_proto::xla::device_assignment_proto::ComputationDevice {
                        replica_device_ids: vec![i64::from(selected_device.id)],
                    },
                ],
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let default_options = default_options.encode_to_vec();
    let raw = client.compile(
        rxla_pjrt::Program::new(format, code),
        compile_options.unwrap_or(&default_options),
    )?;
    Ok(Executable {
        raw,
        client: client.clone(),
        inputs: inputs.to_vec(),
        output_count,
    })
}
impl Tensor {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    fn binary(&self, rhs: &Self, op: Binary) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) {
            return Err(err("cross-graph operands"));
        }
        if self.shape != rhs.shape {
            return Err(err(
                "elementwise shape mismatch; broadcasting must be explicit",
            ));
        }
        self.graph().node(
            Op::Binary(op),
            vec![self.node_id(), rhs.node_id()],
            &self.shape,
        )
    }
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Binary::Add)
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Binary::Mul)
    }
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Binary::Sub)
    }
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Binary::Div)
    }
    pub fn exp(&self) -> Result<Self> {
        self.unary(Unary::Exp)
    }
    /// Matrix multiplication with trailing matrix axes and broadcast batch axes.
    /// Vectors are promoted temporarily; vector × vector returns a scalar.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if !Arc::ptr_eq(&self.graph().0, &rhs.graph().0) {
            return Err(err("cross-graph operands"));
        }
        if self.shape.is_empty() || rhs.shape.is_empty() {
            return Err(err("matmul does not accept scalars"));
        }
        let lhs_vector = self.shape.len() == 1;
        let rhs_vector = rhs.shape.len() == 1;
        let mut lhs_shape = self.shape.to_vec();
        let mut rhs_shape = rhs.shape.to_vec();
        if lhs_vector {
            lhs_shape.insert(0, 1);
        }
        if rhs_vector {
            rhs_shape.push(1);
        }
        let l = lhs_shape.len();
        let r = rhs_shape.len();
        if lhs_shape[l - 1] != rhs_shape[r - 2] {
            return Err(err("matmul contracting dimensions differ"));
        }
        let batch_rank = (l - 2).max(r - 2);
        let mut batch = vec![1; batch_rank];
        for (operand, rank) in [(&lhs_shape, l), (&rhs_shape, r)] {
            for (axis, &dim) in operand[..rank - 2].iter().enumerate() {
                let target = &mut batch[batch_rank - (rank - 2) + axis];
                if *target == 1 {
                    *target = dim;
                } else if dim != 1 && dim != *target {
                    return Err(err("matmul batch dimensions cannot broadcast"));
                }
            }
        }
        let mut lhs_target = batch.clone();
        lhs_target.extend_from_slice(&lhs_shape[l - 2..]);
        let mut rhs_target = batch.clone();
        rhs_target.extend_from_slice(&rhs_shape[r - 2..]);
        let lhs = if lhs_vector {
            self.reshape(&lhs_shape)?
        } else {
            self.clone()
        };
        let rhs = if rhs_vector {
            rhs.reshape(&rhs_shape)?
        } else {
            rhs.clone()
        };
        let lhs = lhs.broadcast_to(&lhs_target)?;
        let rhs = rhs.broadcast_to(&rhs_target)?;
        let mut output = batch;
        output.extend_from_slice(&[lhs_shape[l - 2], rhs_shape[r - 1]]);
        let result = self.graph().node(
            Op::Matmul { batch_rank },
            vec![lhs.node_id(), rhs.node_id()],
            &output,
        )?;
        if rhs_vector {
            output.pop();
        }
        if lhs_vector {
            output.remove(batch_rank);
        }
        if lhs_vector || rhs_vector {
            result.reshape(&output)
        } else {
            Ok(result)
        }
    }
    pub fn reshape(&self, dims: &[i64]) -> Result<Self> {
        if elements(dims)? != elements(&self.shape)? {
            return Err(err("reshape changes element count"));
        }
        self.graph().node(Op::Reshape, vec![self.node_id()], dims)
    }
}
/// Borrowed description of one executable input, in parameter registration order.
/// Inspecting this does not call the plugin or allocate/copy tensor data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSpec<'a> {
    pub shape: &'a [i64],
    pub dtype: DType,
    pub dynamic_bounds: &'a [i64],
}

impl InputSpec<'_> {
    pub fn bound(&self, axis: usize) -> Option<i64> {
        self.dynamic_bounds
            .get(axis)
            .copied()
            .filter(|&bound| bound >= 0)
    }
}

/// Borrowed static description of one ordered tensor result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputSpec<'a> {
    pub shape: &'a [i64],
    pub dtype: DType,
    pub dynamic_bounds: &'a [i64],
}

pub struct Executable {
    raw: rxla_pjrt::Executable,
    client: Client,
    inputs: Vec<TensorType>,
    output_count: usize,
}
impl Executable {
    /// Client that owns this executable and all compatible input buffers.
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn device_count(&self) -> usize {
        self.raw.device_count()
    }

    /// Number of required inputs, including declared but unused parameters.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }
    /// Inspect a parameter without exposing mutable signature metadata.
    /// Returns None for an out-of-range index. Metadata also survives native
    /// artifact restoration through `deserialize_with_metadata`.
    pub fn input_spec(&self, index: usize) -> Option<InputSpec<'_>> {
        self.inputs.get(index).map(|ty| InputSpec {
            shape: &ty.dims,
            dtype: ty.dtype,
            dynamic_bounds: &ty.dynamic_bounds,
        })
    }
    /// Number of top-level tensor results. Shapes can be queried on returned
    /// buffers; this is not a state-schema or nested-tuple introspection API.
    pub fn output_count(&self) -> usize {
        self.output_count
    }
    /// Retrieve the backend's optimized HLO for diagnostics. This is compiler IR,
    /// not a portable/native executable cache artifact or a profiling measurement.
    pub fn optimized_hlo_proto(&self) -> Result<HloModuleProto> {
        decode_optimized_hlo(self.raw.optimized_program("hlo")?)
    }

    /// Export backend-specific executable bytes. Reloading uses the unsafe
    /// low-level Client::deserialize_executable API and requires a trusted artifact.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.raw.serialize()?)
    }
    /// Convenience host execution; use resident buffers through `execute` for hot paths.
    pub fn run(&self, inputs: &[&[f32]]) -> Result<Vec<f32>> {
        if self.output_count != 1 {
            return Err(Error::SingleOutputRequired {
                actual: self.output_count,
            });
        }
        let mut outputs = self.run_many(inputs)?;
        Ok(outputs.remove(0))
    }
    pub fn run_many(&self, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if let Some((index, input)) = self
            .inputs
            .iter()
            .enumerate()
            .find(|(_, input)| input.dtype != DType::F32)
        {
            return Err(Error::F32InputRequired {
                index,
                dtype: input.dtype,
            });
        }
        if inputs.len() != self.inputs.len() {
            return Err(Error::ExecutableInputCount {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        let buffers: Vec<_> = inputs
            .iter()
            .zip(&self.inputs)
            .map(|(v, s)| Ok(self.client.buffer(&s.dims, v)?))
            .collect::<Result<_>>()?;
        let refs: Vec<_> = buffers.iter().collect();
        let outputs = self.raw.execute(&refs)?;
        outputs
            .iter()
            .map(|buffer| Ok(buffer.to_vec::<f32>()?))
            .collect()
    }
    pub fn execute(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        self.validate_inputs(inputs)?;
        Ok(self.raw.execute(inputs)?)
    }
    /// Replicate complete logical inputs across every executable device, run a
    /// multi-device SPMD executable, and return the first replica's complete
    /// logical outputs. This requires replicated program boundaries; it is the
    /// bridge used while auto-sharding only partitions internal values.
    pub fn execute_replicated(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        self.validate_inputs(inputs)?;
        let device_count = self.raw.device_count();
        if device_count == 1 {
            return Ok(self.raw.execute(inputs)?);
        }
        let mut per_device = Vec::with_capacity(device_count);
        for device_index in 0..device_count {
            let mut device_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                device_inputs.push(input.copy_to_device_via_host(&self.client, device_index)?);
            }
            per_device.push(device_inputs);
        }
        let references = per_device
            .iter()
            .map(|values| values.iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let lists = references.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut outputs = self.raw.execute_sharded(&lists)?;
        if let Some((device, values)) = outputs
            .iter()
            .enumerate()
            .find(|(_, values)| values.len() != self.output_count)
        {
            return Err(Error::ShardedOutputCount {
                device,
                expected: self.output_count,
                actual: values.len(),
            });
        }
        Ok(outputs.remove(0))
    }
    /// Validate the input signature and submit without explicitly waiting for
    /// device completion. Native owners are retained by the returned handle;
    /// callers may drop the original inputs/executable. Submission may block in
    /// the plugin. `wait()` returns results; dropping the handle waits and discards
    /// results/errors. No Future, cancellation, donation or kernel-overlap promise.
    /// Stateful Session::run remains synchronous; no state is committed here.
    pub fn submit(&self, inputs: &[&Buffer]) -> Result<PendingExecution> {
        self.validate_inputs(inputs)?;
        Ok(self.raw.submit(inputs)?)
    }
    fn validate_inputs(&self, inputs: &[&Buffer]) -> Result<()> {
        if inputs.len() != self.inputs.len() {
            return Err(Error::ExecutableInputCount {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        for (index, (buffer, expected)) in inputs.iter().zip(&self.inputs).enumerate() {
            let actual = buffer.dimensions()?;
            let shape_matches = actual.len() == expected.dims.len()
                && actual.iter().enumerate().all(|(axis, &dimension)| {
                    if expected.dims[axis] == -1 {
                        expected
                            .bound(axis)
                            .is_some_and(|bound| dimension >= 0 && dimension <= bound)
                    } else {
                        dimension == expected.dims[axis]
                    }
                });
            if !shape_matches {
                return Err(Error::ExecutableInputShape {
                    index,
                    expected: expected.dims.clone(),
                    actual,
                });
            }
            let expected_dtype = expected.dtype;
            let actual_dtype = buffer.dtype()?;
            if actual_dtype != expected_dtype {
                return Err(Error::ExecutableInputDType {
                    index,
                    expected: expected_dtype,
                    actual: actual_dtype,
                });
            }
        }
        Ok(())
    }
}

fn decode_optimized_hlo(program: rxla_pjrt::OptimizedProgram) -> Result<HloModuleProto> {
    match program.format.as_str() {
        "hlo" => HloModuleProto::decode(program.code.as_slice()).map_err(|source| {
            Error::OptimizedProgramDecode {
                format: program.format,
                source,
            }
        }),
        "hlo_with_config" => {
            rxla_xla_proto::xla::HloModuleProtoWithConfig::decode(program.code.as_slice())
                .map_err(|source| Error::OptimizedProgramDecode {
                    format: program.format,
                    source,
                })?
                .hlo_module
                .ok_or(Error::MissingOptimizedHloModule)
        }
        _ => Err(Error::UnsupportedOptimizedProgram {
            format: program.format,
        }),
    }
}

#[cfg(test)]
mod optimized_program_tests {
    use super::*;

    #[test]
    fn optimized_program_format_decoding_is_explicit() {
        let module = HloModuleProto {
            name: "diagnostic".into(),
            ..Default::default()
        };
        let program = |format: &str, code: Vec<u8>| rxla_pjrt::OptimizedProgram {
            format: format.into(),
            code,
        };
        assert_eq!(
            decode_optimized_hlo(program("hlo", module.encode_to_vec())).unwrap(),
            module
        );
        let wrapped = rxla_xla_proto::xla::HloModuleProtoWithConfig {
            hlo_module: Some(module.clone()),
            config: None,
        };
        assert_eq!(
            decode_optimized_hlo(program("hlo_with_config", wrapped.encode_to_vec())).unwrap(),
            module
        );
        assert!(matches!(
            decode_optimized_hlo(program("hlo_with_config", vec![])),
            Err(Error::MissingOptimizedHloModule)
        ));
        assert!(matches!(
            decode_optimized_hlo(program("unknown", module.encode_to_vec())),
            Err(Error::UnsupportedOptimizedProgram { format }) if format == "unknown"
        ));
        for format in ["hlo", "hlo_with_config"] {
            assert!(matches!(
                decode_optimized_hlo(program(format, vec![0xff])),
                Err(Error::OptimizedProgramDecode {
                    format: actual,
                    ..
                }) if actual == format
            ));
        }
    }
}
