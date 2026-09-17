//! Scoped parameter effects for tensor tracing.
//!
//! A [`Cx`] is deliberately not a global variable store. [`Model`] traces
//! parameter effects once into both an immutable schema and executable IR, so
//! model code has no hidden call-order state or schema-discovery replay.

use rxla_core::{DType, StateGraph, StateSlot, Tensor};
use snafu::{OptionExt, Snafu, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

mod applied;
pub use applied::{
    AppliedModel, BoundModel, CompiledModel, CompiledModelSessionBuilder, CompiledStatefulModel,
    ModelArguments, ModelSession, ModelSessionBuffers, ModelSessionBuilder, TransformState,
};
mod inputs;
pub use inputs::{ModelHandler, ModelInput, ModelInputValues, ModelInputs};
mod initializer;
pub use initializer::Initializer;
mod layers;
pub use layers::{
    BatchNorm, Conv2d, Embedding, GroupNorm, ImageLayout, Layer, LayerNorm, Linear, NamedLayer,
    QuantizedLinear, RmsNorm, TensorApply,
};
mod model_path;
pub use model_path::{ModelPath, ModelPathSegment};

/// Build a validated model-tree cursor by pushing each path segment in order.
#[macro_export]
macro_rules! path {
    ($root:ident $(/ $segment:tt)+ $(/)?) => {{
        (|| -> $crate::Result<$crate::Cx> {
            let mut path = $root.clone();
            $(path.__push_model_path_segment($segment)?;)+
            Ok(path)
        })()
    }};
}
mod outputs;
pub use outputs::{ModelOutputValues, ModelOutputs};
mod schema;
pub use schema::{ModelArgument, ModelInputSpec, ModelSchema, ParameterSpec, StateSpec};
mod selection;
pub use selection::{ParameterId, ParameterSelection};

/// A reusable effect-based model definition.
///
/// It captures one ordinary Rust function and traces schema plus executable IR
/// in one invocation. Compiled execution artifacts use
/// [`rxla_core::Program`]; keeping the names distinct avoids import aliases in
/// applications that construct and run models in the same module.
pub struct Model<F, I = NoModelInputs> {
    apply: F,
    inputs: I,
    mode: ExecutionMode,
}

/// Execution semantics selected once for a complete model trace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    Training,
    #[default]
    Inference,
}

/// Marker used by model functions that declare any inputs themselves.
pub struct NoModelInputs;

/// Recoverable parameter-effect and model-binding failures.
#[derive(Debug, Snafu)]
#[non_exhaustive]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(transparent)]
    Tensor { source: rxla_core::Error },
    #[snafu(transparent)]
    Pjrt { source: rxla_core::PjrtError },
    #[snafu(display("invalid model name segment {name:?}: expected nonempty and dot-free"))]
    InvalidName { name: String },
    #[snafu(display("parameter {path:?} has a negative dimension"))]
    NegativeParameterDimension { path: String },
    #[snafu(display("parameter dtype {dtype:?} is unsupported"))]
    UnsupportedParameterDType { dtype: DType },
    #[snafu(display(
        "parameter storage dtype {storage_dtype:?} cannot be converted to compute dtype {compute_dtype:?}"
    ))]
    UnsupportedParameterDTypeConversion {
        storage_dtype: DType,
        compute_dtype: DType,
    },
    #[snafu(display("resident parameter {path:?} has no session initializer"))]
    MissingResidentParameterInitializer { path: String },
    #[snafu(display(
        "source session contains resident parameter {path:?} absent from the target model"
    ))]
    UnexpectedResidentParameter { path: String },
    #[snafu(display("parameter {path:?} is not resident in this model trace"))]
    ParameterNotResident { path: String },
    #[snafu(display("expected {expected} resident parameter updates, received {actual}"))]
    ParameterUpdateCount { expected: usize, actual: usize },
    #[snafu(display("model input dtype {dtype:?} is unsupported"))]
    UnsupportedInputDType { dtype: DType },
    #[snafu(display("parameter declaration for {path:?} is incompatible with the schema"))]
    IncompatibleParameter { path: String },
    #[snafu(display("parameter {path:?} is absent from the schema"))]
    UnknownParameter { path: String },
    #[snafu(display("duplicate parameter binding {path:?}"))]
    DuplicateBinding { path: String },
    #[snafu(display("missing parameter binding {path:?}"))]
    MissingBinding { path: String },
    #[snafu(display("model expected {expected} inputs, received {actual}"))]
    InputCount { expected: usize, actual: usize },
    #[snafu(display("model expected {expected} bound parameters, received {actual}"))]
    ParameterCount { expected: usize, actual: usize },
    #[snafu(display("parameter selection belongs to a different model schema"))]
    SelectionSchemaMismatch,
    #[snafu(display("traced model structure does not match the supplied schema"))]
    ModelSchemaMismatch,
    #[snafu(display("{kind} {identity}: shape does not match the schema"))]
    BufferShape {
        kind: &'static str,
        identity: String,
    },
    #[snafu(display("{kind} {identity}: dtype does not match the schema"))]
    BufferDType {
        kind: &'static str,
        identity: String,
    },
    #[snafu(display("{kind} {identity}: buffer belongs to another PJRT client"))]
    BufferClient {
        kind: &'static str,
        identity: String,
    },
    #[snafu(display("{layer} requires {requirement}"))]
    InvalidLayerInput {
        layer: &'static str,
        requirement: &'static str,
    },
    #[snafu(display("state handle belongs to another model trace"))]
    ForeignState,
    #[snafu(display("{effect} is not supported inside a conditional branch yet"))]
    ConditionalEffect { effect: &'static str },
    #[snafu(display("{operation} requires a stateless model"))]
    StatefulOperation { operation: &'static str },
    #[snafu(display("resident parameter {path:?} must be initialized through a session"))]
    ResidentParameterBinding { path: String },
    #[snafu(display("unknown session state {path:?}"))]
    UnknownState { path: String },
    #[snafu(display("duplicate session state initializer {path:?}"))]
    DuplicateStateInitializer { path: String },
    #[snafu(display("duplicate resident parameter initializer {path:?}"))]
    DuplicateResidentParameterInitializer { path: String },
    #[snafu(display("unknown RNG stream {name:?}"))]
    UnknownRng { name: String },
    #[snafu(display("session is missing {kind} {path:?}"))]
    MissingSessionValue { kind: &'static str, path: String },
    #[snafu(display("session contains state absent from its applied model"))]
    UnexpectedSessionState,
    #[snafu(display("session contains unused {kind} initializers"))]
    UnusedSessionInitializers { kind: &'static str },
    #[snafu(display("invalid {operation} probability {probability}: expected {requirement}"))]
    InvalidProbability {
        operation: &'static str,
        probability: f32,
        requirement: &'static str,
    },
    #[snafu(display("state declaration for {path:?} is incompatible with the schema"))]
    IncompatibleState { path: String },
    #[snafu(display("transform state {path:?} is already declared"))]
    DuplicateTransformState { path: String },
    #[snafu(display("model returned {actual} output buffers, expected {expected}"))]
    OutputCount { expected: usize, actual: usize },
    #[snafu(display("output buffers do not match the requested result structure"))]
    OutputStructure,
    #[snafu(display("parameter {path:?} has no initializer"))]
    MissingInitializer { path: String },
    #[snafu(display("invalid initializer for {path:?}: {message}"))]
    InvalidInitializer { path: String, message: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl<F> Model<F, NoModelInputs> {
    /// Capture one model `apply` function for repeated effect interpretation.
    pub fn new(apply: F) -> Self {
        Self {
            apply,
            inputs: NoModelInputs,
            mode: ExecutionMode::Inference,
        }
    }

    /// Supply structured input declarations to an `apply(cx, inputs)` function.
    pub fn inputs<I>(self, inputs: I) -> Model<F, I> {
        Model {
            apply: self.apply,
            inputs,
            mode: self.mode,
        }
    }
}

impl<F, I> Model<F, I> {
    pub fn training(mut self) -> Self {
        self.mode = ExecutionMode::Training;
        self
    }

    pub fn inference(mut self) -> Self {
        self.mode = ExecutionMode::Inference;
        self
    }
}

impl<F, I> Model<F, I>
where
    I: ModelInputs,
{
    /// Discover parameter/input effects and produce the executable model trace.
    ///
    /// The frozen declarations are available through [`AppliedModel::schema`].
    /// Zero-input and structured-input models use this same interpretation path.
    pub fn trace<Marker>(
        &self,
    ) -> std::result::Result<AppliedModel, <F as ModelHandler<I, Marker>>::Error>
    where
        F: ModelHandler<I, Marker>,
    {
        trace_once_mode(self.mode, |cx| self.apply.invoke(cx.clone(), &self.inputs))
    }

    /// Trace typed inputs once with every parameter stored as resident state.
    pub fn trace_resident_all<Marker>(
        &self,
    ) -> std::result::Result<
        (ParameterSelection, AppliedModel),
        <F as ModelHandler<I, Marker>>::Error,
    >
    where
        F: ModelHandler<I, Marker>,
    {
        let applied =
            trace_once_resident_all(self.mode, |cx| self.apply.invoke(cx.clone(), &self.inputs))?;
        let selection = applied.schema().select_all();
        Ok((selection, applied))
    }

    /// Trace typed inputs once with one lexical parameter scope resident.
    pub fn trace_resident_under<Marker>(
        &self,
        scope: &str,
    ) -> std::result::Result<
        (ParameterSelection, AppliedModel),
        <F as ModelHandler<I, Marker>>::Error,
    >
    where
        F: ModelHandler<I, Marker>,
    {
        let applied = trace_once_resident_under(scope, self.mode, |cx| {
            self.apply.invoke(cx.clone(), &self.inputs)
        })?;
        let selection = applied.schema().select_under(scope);
        Ok((selection, applied))
    }

    /// Trace once with resident parameters selected from an existing schema.
    ///
    /// This supports distinct training and inference functions that declare the
    /// same model structure. The selection carries its schema provenance; the
    /// newly traced structure is compared with that schema before it is returned.
    pub fn trace_resident<Marker>(
        &self,
        selection: &ParameterSelection,
    ) -> std::result::Result<AppliedModel, <F as ModelHandler<I, Marker>>::Error>
    where
        F: ModelHandler<I, Marker>,
    {
        let paths = selection
            .parameters()
            .map(|(_, parameter)| parameter.path().to_owned())
            .collect();
        let mut applied = trace_once_resident_selected(paths, self.mode, |cx| {
            self.apply.invoke(cx.clone(), &self.inputs)
        })?;
        if applied.schema() != selection.schema() {
            return Err(<F as ModelHandler<I, Marker>>::Error::from(
                Error::ModelSchemaMismatch,
            ));
        }
        applied.adopt_schema_identity(selection.schema());
        Ok(applied)
    }
}

enum InitResidency {
    None,
    All,
    Under(String),
    Selected(BTreeSet<String>),
}

impl InitResidency {
    fn contains(&self, path: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Under(scope) => selection::path_is_under(path, scope),
            Self::Selected(paths) => paths.contains(path),
        }
    }
}

/// The explicit interpreter for scoped parameter effects.
///
/// A model function receives this context once while [`Model::trace`] records
/// its parameter, input, state, RNG, and tensor operations.
#[derive(Clone)]
pub struct Cx {
    inner: Arc<Mutex<CxInner>>,
    path: ModelPath,
}

struct CxInner {
    graph: StateGraph,
    schema: ModelSchema,
    parameter_values: BTreeMap<String, Tensor>,
    residency: InitResidency,
    states: BTreeMap<String, StateDeclaration>,
    rngs: BTreeMap<String, RngStream>,
    resident_parameters: BTreeMap<String, StateSlot>,
    conditional_depth: usize,
    mode: ExecutionMode,
}

struct StateDeclaration {
    slot: StateSlot,
    shape: Vec<i64>,
    dtype: DType,
    initializer: Initializer,
}

/// Stable identity of a named resident value in a model trace.
#[derive(Clone)]
pub struct State {
    path: String,
    slot: StateSlot,
}

struct RngStream {
    key: [Tensor; 2],
    counters: [State; 2],
    next: [Tensor; 2],
    available: Tensor,
}

/// A named counter-based device RNG effect in the unified model context.
pub struct Rng {
    cx: Cx,
    path: String,
}

impl Cx {
    #[cfg(test)]
    fn init() -> Self {
        Self::init_with_residency(InitResidency::None, ExecutionMode::Training)
    }

    fn init_resident_all(mode: ExecutionMode) -> Self {
        Self::init_with_residency(InitResidency::All, mode)
    }

    fn init_resident_under(scope: &str, mode: ExecutionMode) -> Self {
        Self::init_with_residency(InitResidency::Under(scope.to_owned()), mode)
    }

    fn init_resident_selected(paths: BTreeSet<String>, mode: ExecutionMode) -> Self {
        Self::init_with_residency(InitResidency::Selected(paths), mode)
    }

    fn init_with_residency(residency: InitResidency, mode: ExecutionMode) -> Self {
        Self {
            path: ModelPath::root(),
            inner: Arc::new(Mutex::new(CxInner {
                graph: StateGraph::default(),
                schema: ModelSchema::default(),
                parameter_values: BTreeMap::new(),
                residency,
                states: BTreeMap::new(),
                rngs: BTreeMap::new(),
                resident_parameters: BTreeMap::new(),
                conditional_depth: 0,
                mode,
            })),
        }
    }

    pub fn mode(&self) -> ExecutionMode {
        self.inner.lock().expect("trace context lock poisoned").mode
    }

    /// Declare/read an F32 parameter at the current lexical scope.
    pub fn param(&self, name: &str, shape: &[i64]) -> Result<Tensor> {
        self.param_dtype(name, shape, DType::F32)
    }

    /// Declare/read an F32 parameter with a deterministic initialization policy.
    pub fn param_initialized(
        &self,
        name: &str,
        shape: &[i64],
        initializer: Initializer,
    ) -> Result<Tensor> {
        self.param_dtype_initialized(name, shape, DType::F32, Some(initializer))
    }

    /// Declare/read a parameter with its storage dtype.
    ///
    /// The current symbolic tensor surface supports F32 parameters, frozen
    /// BF16 storage exposed as F32 computation values, and raw U8 storage for
    /// explicitly dequantized inference layers. Other dtypes are rejected.
    pub fn param_dtype(&self, name: &str, shape: &[i64], dtype: DType) -> Result<Tensor> {
        let compute_dtype = match dtype {
            DType::F16 | DType::BF16 => DType::F32,
            _ => dtype,
        };
        self.param_with_dtypes_impl(name, shape, dtype, compute_dtype, None)
    }

    /// Declare/read a parameter with independent checkpoint and computation
    /// element types. Any conversion is represented explicitly in the IR.
    pub fn param_with_dtypes(
        &self,
        name: &str,
        shape: &[i64],
        storage_dtype: DType,
        compute_dtype: DType,
    ) -> Result<Tensor> {
        self.param_with_dtypes_impl(name, shape, storage_dtype, compute_dtype, None)
    }

    /// Declare/read a mixed-precision parameter with an initialization policy
    /// applied in its checkpoint storage type.
    pub fn param_with_dtypes_initialized(
        &self,
        name: &str,
        shape: &[i64],
        storage_dtype: DType,
        compute_dtype: DType,
        initializer: Initializer,
    ) -> Result<Tensor> {
        self.param_with_dtypes_impl(name, shape, storage_dtype, compute_dtype, Some(initializer))
    }

    fn param_dtype_initialized(
        &self,
        name: &str,
        shape: &[i64],
        dtype: DType,
        initializer: Option<Initializer>,
    ) -> Result<Tensor> {
        let compute_dtype = match dtype {
            DType::F16 | DType::BF16 => DType::F32,
            _ => dtype,
        };
        self.param_with_dtypes_impl(name, shape, dtype, compute_dtype, initializer)
    }

    fn param_with_dtypes_impl(
        &self,
        name: &str,
        shape: &[i64],
        storage_dtype: DType,
        compute_dtype: DType,
        initializer: Option<Initializer>,
    ) -> Result<Tensor> {
        validate_name(name)?;
        let path = self.path(name);
        ensure!(
            shape.iter().all(|&dim| dim >= 0),
            NegativeParameterDimensionSnafu { path }
        );
        if let Some(initializer) = initializer {
            initializer.validate(&path, shape, storage_dtype)?;
        }
        let requested = ParameterSpec {
            path: path.clone(),
            shape: shape.to_vec(),
            storage_dtype,
            compute_dtype,
            initializer,
        };
        let mut inner = self.inner.lock().expect("trace context lock poisoned");
        let CxInner {
            graph,
            schema,
            parameter_values,
            residency,
            resident_parameters,
            ..
        } = &mut *inner;
        if let Some(existing) = schema.get(&path) {
            ensure!(existing == &requested, IncompatibleParameterSnafu { path });
            return Ok(parameter_values
                .get(&path)
                .expect("schema and parameter value are inserted together")
                .clone());
        }
        let value = if residency.contains(&path) {
            let (value, slot) =
                resident_parameter_tensor(graph, &path, shape, storage_dtype, compute_dtype)?;
            resident_parameters.insert(path.clone(), slot);
            value
        } else {
            parameter_tensor(graph, shape, storage_dtype, compute_dtype)?
        };
        schema.push_parameter(requested);
        parameter_values.insert(path, value.clone());
        Ok(value)
    }

    /// Derive an independent handle for a child lexical effect scope.
    pub fn scope(&self, name: impl Into<String>) -> Result<Self> {
        self.at(name.into())
    }

    /// Enter a repeated block path such as `blocks.17` without formatting it at
    /// every model call site.
    pub fn scope_index(&self, collection: &str, index: usize) -> Result<Self> {
        let mut child = self.clone();
        child.__push_model_path_segment(collection)?;
        child.__push_model_path_segment(index)?;
        Ok(child)
    }

    /// Build a structured conditional while threading this model's parameter
    /// effect context through both branches. Parameters declared in either
    /// branch are hoisted into the model ABI and captured by the branch region.
    pub fn cond<Then, Else>(
        &self,
        predicate: &Tensor,
        then_branch: Then,
        else_branch: Else,
    ) -> Result<Tensor>
    where
        Then: FnOnce(Cx) -> Result<Tensor>,
        Else: FnOnce(Cx) -> Result<Tensor>,
    {
        self.cond_many(
            predicate,
            |cx| then_branch(cx).map(|value| vec![value]),
            |cx| else_branch(cx).map(|value| vec![value]),
        )
        .map(|mut values| values.remove(0))
    }

    /// Multi-result conditional that also treats resident state versions as
    /// hidden region results. Each branch starts from the same state snapshot;
    /// the selected versions become visible to subsequent model operations.
    pub fn cond_many<Then, Else>(
        &self,
        predicate: &Tensor,
        then_branch: Then,
        else_branch: Else,
    ) -> Result<Vec<Tensor>>
    where
        Then: FnOnce(Cx) -> Result<Vec<Tensor>>,
        Else: FnOnce(Cx) -> Result<Vec<Tensor>>,
    {
        let initial_state = self
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .graph
            .symbolic_state_versions();
        let mut visible_results = None;
        self.inner
            .lock()
            .expect("trace context lock poisoned")
            .conditional_depth += 1;
        let then_cx = self.clone();
        let else_cx = self.clone();
        let result: Result<Vec<Tensor>> = Tensor::cond_many_with(
            predicate,
            &mut (),
            |_| {
                let mut outputs = then_branch(then_cx.clone())?;
                visible_results = Some(outputs.len());
                let mut inner = then_cx.inner.lock().expect("trace context lock poisoned");
                outputs.extend(inner.graph.symbolic_state_versions());
                inner
                    .graph
                    .replace_symbolic_state_versions(&initial_state)?;
                Ok(outputs)
            },
            |_| {
                let mut outputs = else_branch(else_cx.clone())?;
                let mut inner = else_cx.inner.lock().expect("trace context lock poisoned");
                outputs.extend(inner.graph.symbolic_state_versions());
                inner
                    .graph
                    .replace_symbolic_state_versions(&initial_state)?;
                Ok(outputs)
            },
        );
        let mut inner = self.inner.lock().expect("trace context lock poisoned");
        inner.conditional_depth -= 1;
        inner
            .graph
            .replace_symbolic_state_versions(&initial_state)?;
        let mut outputs = result?;
        let visible_results = visible_results.unwrap_or(0);
        let merged_state = outputs.split_off(visible_results);
        inner.graph.replace_symbolic_state_versions(&merged_state)?;
        Ok(outputs)
    }

    /// Derive an independent handle extended by several lexical path segments.
    pub fn scope_path<I, S>(&self, segments: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut path = self.path.clone();
        for segment in segments {
            path = path.at(segment.into())?;
        }
        Ok(Self {
            inner: self.inner.clone(),
            path,
        })
    }

    /// Derive a context positioned at an arbitrary relative model path.
    pub fn at<S: ModelPathSegment>(&self, segment: S) -> Result<Self> {
        let mut child = self.clone();
        child.__push_model_path_segment(segment)?;
        Ok(child)
    }

    #[doc(hidden)]
    pub fn __push_model_path_segment<S: ModelPathSegment>(&mut self, segment: S) -> Result<()> {
        self.path.push(segment)?;
        Ok(())
    }

    /// Return the pure coordinate of this context in the model tree.
    pub fn model_path(&self) -> &ModelPath {
        &self.path
    }

    /// Create a visible F32 model input in this trace.
    pub fn input(&self, shape: &[i64]) -> Result<Tensor> {
        self.input_dtype(shape, DType::F32)
    }

    /// Create a visible input with an explicit dtype.
    pub fn input_dtype(&self, shape: &[i64], dtype: DType) -> Result<Tensor> {
        let requested = ModelInputSpec {
            shape: shape.to_vec(),
            dtype,
        };
        let mut inner = self.inner.lock().expect("trace context lock poisoned");
        inner.schema.push_input(requested);
        match dtype {
            DType::F32 | DType::I32 | DType::U8 => {
                Ok(inner.graph.input_with_dtype(shape, dtype)?)
            }
            DType::BF16 => Ok(inner.graph.input_bf16_as_f32(shape)?),
            _ => UnsupportedInputDTypeSnafu { dtype }.fail(),
        }
    }

    /// Create a graph-local I32 coordinate tensor without adding an ABI input.
    pub fn iota_i32(&self, shape: &[i64], axis: usize) -> Result<Tensor> {
        Ok(self
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .graph
            .iota_i32(shape, axis)?)
    }

    /// Create a graph-local F32 constant without adding an ABI input.
    pub fn constant(&self, shape: &[i64], values: &[f32]) -> Result<Tensor> {
        Ok(self
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .graph
            .constant(shape, values)?)
    }

    /// Declare or read named resident state at the current lexical scope.
    pub fn state(&self, name: &str, shape: &[i64], dtype: DType) -> Result<State> {
        self.state_initialized(name, shape, dtype, Initializer::zeros())
    }

    /// Declare resident state with an explicit session initialization policy.
    pub fn state_initialized(
        &self,
        name: &str,
        shape: &[i64],
        dtype: DType,
        initializer: Initializer,
    ) -> Result<State> {
        ensure!(
            self.inner
                .lock()
                .expect("trace context lock poisoned")
                .conditional_depth
                == 0,
            ConditionalEffectSnafu {
                effect: "state declaration"
            }
        );
        validate_name(name)?;
        let path = self.path(name);
        initializer.validate(&path, shape, dtype)?;
        let mut inner = self.inner.lock().expect("trace context lock poisoned");
        if let Some(existing) = inner.states.get(&path) {
            ensure!(
                existing.shape == shape
                    && existing.dtype == dtype
                    && existing.initializer == initializer,
                IncompatibleStateSnafu { path }
            );
            return Ok(State {
                path,
                slot: existing.slot.clone(),
            });
        }
        inner.schema.push_state(StateSpec {
            path: path.clone(),
            shape: shape.to_vec(),
            dtype,
            initializer,
        });
        let slot = inner.graph.state_named(&path, shape, dtype)?;
        inner.states.insert(
            path.clone(),
            StateDeclaration {
                slot: slot.clone(),
                shape: shape.to_vec(),
                dtype,
                initializer,
            },
        );
        Ok(State { path, slot })
    }

    /// Borrow a named Threefry stream. Repeated calls continue the same stream.
    pub fn rng(&self, name: &str) -> Result<Rng> {
        ensure!(
            self.inner
                .lock()
                .expect("trace context lock poisoned")
                .conditional_depth
                == 0,
            ConditionalEffectSnafu { effect: "RNG" }
        );
        validate_name(name)?;
        let path = self.path(name);
        if !self
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .rngs
            .contains_key(&path)
        {
            let words = {
                let scope = self.scope(name)?;
                [
                    scope.state("key0", &[], DType::I32)?,
                    scope.state("key1", &[], DType::I32)?,
                    scope.state("counter_low", &[], DType::I32)?,
                    scope.state("counter_high", &[], DType::I32)?,
                ]
            };
            let values = words
                .iter()
                .map(|word| word.read(self))
                .collect::<Result<Vec<_>>>()?;
            let available = self
                .inner
                .lock()
                .expect("trace context lock poisoned")
                .graph
                .constant(&[], &[1.0])?;
            self.inner
                .lock()
                .expect("trace context lock poisoned")
                .rngs
                .insert(
                    path.clone(),
                    RngStream {
                        key: [values[0].clone(), values[1].clone()],
                        counters: [words[2].clone(), words[3].clone()],
                        next: [values[2].clone(), values[3].clone()],
                        available,
                    },
                );
        }
        Ok(Rng {
            cx: self.clone(),
            path,
        })
    }

    fn validate_state(&self, state: &State) -> Result<()> {
        match self
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .states
            .get(&state.path)
        {
            Some(declaration) if declaration.slot.identity() == state.slot.identity() => Ok(()),
            _ => ForeignStateSnafu.fail(),
        }
    }

    fn finish_rngs(&self) -> Result<()> {
        let mut inner = self.inner.lock().expect("trace context lock poisoned");
        for stream in std::mem::take(&mut inner.rngs).into_values() {
            inner.graph.write_many_if(
                &stream.available,
                &[
                    (&stream.counters[0].slot, stream.next[0].clone()),
                    (&stream.counters[1].slot, stream.next[1].clone()),
                ],
            )?;
        }
        Ok(())
    }

    fn path(&self, name: &str) -> String {
        self.path
            .parameter(name)
            .expect("effect names are validated before paths are constructed")
    }

    fn parameter_tensors(&self) -> Vec<Tensor> {
        let inner = self.inner.lock().expect("trace context lock poisoned");
        inner
            .schema
            .parameters()
            .iter()
            .map(|parameter| {
                inner
                    .parameter_values
                    .get(parameter.path())
                    .expect("each schema parameter has a traced tensor")
                    .clone()
            })
            .collect()
    }

    fn state_slots(&self, schema: &ModelSchema) -> Vec<(String, StateSlot)> {
        let inner = self.inner.lock().expect("trace context lock poisoned");
        schema
            .states()
            .iter()
            .map(|state| {
                (
                    state.path().to_owned(),
                    inner.states[state.path()].slot.clone(),
                )
            })
            .collect()
    }

    fn resident_parameter_slots(&self, schema: &ModelSchema) -> Vec<Option<StateSlot>> {
        let inner = self.inner.lock().expect("trace context lock poisoned");
        schema
            .parameters()
            .iter()
            .map(|parameter| inner.resident_parameters.get(parameter.path()).cloned())
            .collect()
    }

    #[cfg(test)]
    fn into_schema(self) -> ModelSchema {
        self.inner
            .lock()
            .expect("trace context lock poisoned")
            .schema
            .clone()
    }
}

impl State {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn read(&self, cx: &Cx) -> Result<Tensor> {
        cx.validate_state(self)?;
        Ok(cx
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .graph
            .read(&self.slot)?)
    }

    pub fn write(&self, cx: &Cx, value: &Tensor) -> Result<()> {
        cx.validate_state(self)?;
        Ok(cx
            .inner
            .lock()
            .expect("trace context lock poisoned")
            .graph
            .write(&self.slot, value)?)
    }

    pub fn add_(&self, cx: &Cx, value: &Tensor) -> Result<()> {
        let next = self.read(cx)?.add(value)?;
        self.write(cx, &next)
    }
}

impl Rng {
    fn draw(&mut self, shape: &[i64]) -> Result<rxla_core::random::ThreefryBlocks> {
        let (key, next) = {
            let inner = self.cx.inner.lock().expect("trace context lock poisoned");
            let stream = &inner.rngs[&self.path];
            (stream.key.clone(), stream.next.clone())
        };
        let draw = rxla_core::random::threefry2x32_blocks(
            [&key[0], &key[1]],
            [&next[0], &next[1]],
            shape,
        )?;
        let mut inner = self.cx.inner.lock().expect("trace context lock poisoned");
        let stream = inner
            .rngs
            .get_mut(&self.path)
            .expect("RNG handle owns its context");
        stream.available = stream
            .available
            .mul(&draw.counter_wrapped.neg()?.add_scalar(1.0)?)?;
        stream.next = draw.next_counter.clone();
        Ok(draw)
    }

    pub fn blocks(&mut self, shape: &[i64]) -> Result<[Tensor; 2]> {
        Ok(self.draw(shape)?.bits)
    }

    pub fn uniform_f32(&mut self, shape: &[i64]) -> Result<Tensor> {
        let bits = self.draw(shape)?.bits;
        Ok(rxla_core::random::uniform_f32_from_bits(&bits[0])?)
    }

    pub fn normal_f32(&mut self, shape: &[i64]) -> Result<Tensor> {
        let bits = self.draw(shape)?.bits;
        Ok(rxla_core::random::normal_f32_from_bits([
            &bits[0], &bits[1],
        ])?)
    }

    pub fn bernoulli(&mut self, shape: &[i64], probability: f32) -> Result<Tensor> {
        ensure!(
            (0.0..=1.0).contains(&probability),
            InvalidProbabilitySnafu {
                operation: "Bernoulli",
                probability,
                requirement: "a finite value in [0, 1]"
            }
        );
        if probability == 0.0 || probability == 1.0 {
            let key = self
                .cx
                .inner
                .lock()
                .expect("trace context lock poisoned")
                .rngs[&self.path]
                .key[0]
                .clone();
            return Ok(key.scalar(probability)?.broadcast_to(shape)?);
        }
        let uniform = self.uniform_f32(shape)?;
        Ok(uniform.lt_mask(&uniform.scalar(probability)?.broadcast_to(shape)?)?)
    }

    pub fn dropout(
        &mut self,
        input: &Tensor,
        keep_probability: f32,
    ) -> Result<rxla_core::random::DropoutSample> {
        ensure!(
            keep_probability.is_finite() && keep_probability > 0.0 && keep_probability <= 1.0,
            InvalidProbabilitySnafu {
                operation: "dropout keep",
                probability: keep_probability,
                requirement: "a finite value in (0, 1]"
            }
        );
        let keep_mask = self.bernoulli(input.shape(), keep_probability)?;
        let output = input.dropout_with_mask(&keep_mask, keep_probability)?;
        Ok(rxla_core::random::DropoutSample { output, keep_mask })
    }
}

fn parameter_tensor(
    graph: &mut StateGraph,
    shape: &[i64],
    storage_dtype: DType,
    compute_dtype: DType,
) -> Result<Tensor> {
    validate_parameter_dtypes(storage_dtype, compute_dtype)?;
    let stored = graph.input_with_dtype(shape, storage_dtype)?;
    if storage_dtype == compute_dtype {
        Ok(stored)
    } else {
        Ok(stored.cast(compute_dtype)?)
    }
}

fn resident_parameter_tensor(
    graph: &mut StateGraph,
    path: &str,
    shape: &[i64],
    storage_dtype: DType,
    compute_dtype: DType,
) -> Result<(Tensor, StateSlot)> {
    validate_parameter_dtypes(storage_dtype, compute_dtype)?;
    let slot = graph.state_named(&format!("__parameter.{path}"), shape, storage_dtype)?;
    let stored = graph.read(&slot)?;
    let value = if storage_dtype == compute_dtype {
        stored
    } else {
        stored.cast(compute_dtype)?
    };
    Ok((value, slot))
}

fn validate_parameter_dtypes(storage_dtype: DType, compute_dtype: DType) -> Result<()> {
    let storage_supported = matches!(
        storage_dtype,
        DType::U8 | DType::F16 | DType::BF16 | DType::F32
    );
    let compute_supported = matches!(
        compute_dtype,
        DType::U8 | DType::F16 | DType::BF16 | DType::F32
    );
    if !storage_supported {
        return UnsupportedParameterDTypeSnafu {
            dtype: storage_dtype,
        }
        .fail();
    }
    if !compute_supported {
        return UnsupportedParameterDTypeSnafu {
            dtype: compute_dtype,
        }
        .fail();
    }
    if storage_dtype != compute_dtype
        && !matches!(compute_dtype, DType::F16 | DType::BF16 | DType::F32)
    {
        return UnsupportedParameterDTypeConversionSnafu {
            storage_dtype,
            compute_dtype,
        }
        .fail();
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && !name.contains('.'),
        InvalidNameSnafu { name }
    );
    Ok(())
}

/// Interpret parameter effects as declarations and return the frozen schema.
#[cfg(test)]
fn init_with_error<T, E>(
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<(ModelSchema, T), E>
where
    E: From<Error>,
{
    let cx = Cx::init();
    let result = body(cx.clone())?;
    cx.finish_rngs().map_err(E::from)?;
    Ok((cx.into_schema(), result))
}

#[cfg(test)]
fn init<T>(body: impl FnOnce(Cx) -> Result<T>) -> Result<(ModelSchema, T)> {
    init_with_error(body)
}

#[cfg(test)]
fn apply<T: ModelOutputs>(
    schema: &ModelSchema,
    body: impl FnOnce(Cx) -> Result<T>,
) -> Result<AppliedModel> {
    let applied = trace_once(body)?;
    assert_eq!(applied.schema(), schema, "test replay changed model schema");
    Ok(applied)
}

#[cfg(test)]
fn trace_once<T, E>(
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<AppliedModel, E>
where
    T: ModelOutputs,
    E: From<Error>,
{
    trace_once_with(Cx::init(), body)
}

fn trace_once_mode<T, E>(
    mode: ExecutionMode,
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<AppliedModel, E>
where
    T: ModelOutputs,
    E: From<Error>,
{
    trace_once_with(Cx::init_with_residency(InitResidency::None, mode), body)
}

fn trace_once_resident_all<T, E>(
    mode: ExecutionMode,
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<AppliedModel, E>
where
    T: ModelOutputs,
    E: From<Error>,
{
    trace_once_with(Cx::init_resident_all(mode), body)
}

fn trace_once_resident_under<T, E>(
    scope: &str,
    mode: ExecutionMode,
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<AppliedModel, E>
where
    T: ModelOutputs,
    E: From<Error>,
{
    trace_once_with(Cx::init_resident_under(scope, mode), body)
}

fn trace_once_resident_selected<T, E>(
    paths: BTreeSet<String>,
    mode: ExecutionMode,
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<AppliedModel, E>
where
    T: ModelOutputs,
    E: From<Error>,
{
    trace_once_with(Cx::init_resident_selected(paths, mode), body)
}

fn trace_once_with<T, E>(
    cx: Cx,
    body: impl FnOnce(Cx) -> std::result::Result<T, E>,
) -> std::result::Result<AppliedModel, E>
where
    T: ModelOutputs,
    E: From<Error>,
{
    let outputs = body(cx.clone())?.into_tensors();
    cx.finish_rngs().map_err(E::from)?;
    let schema = cx
        .inner
        .lock()
        .expect("trace context lock poisoned")
        .schema
        .clone();
    let parameters = cx.parameter_tensors();
    let states = cx.state_slots(&schema);
    let resident_parameters = cx.resident_parameter_slots(&schema);
    let graph = cx
        .inner
        .lock()
        .expect("trace context lock poisoned")
        .graph
        .clone();
    Ok(AppliedModel::new(
        graph,
        states,
        outputs,
        parameters,
        resident_parameters,
        schema,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{Buffer, CacheLimits, Client, ClientOptions, Compiler, Conv2dOptions};
    use std::cell::Cell;

    #[test]
    fn model_execution_mode_is_selected_once_for_the_trace() {
        let training_seen = Cell::new(None);
        Model::new(|cx: Cx| -> Result<Tensor> {
            training_seen.set(Some(cx.mode()));
            cx.input(&[1])
        })
        .training()
        .trace()
        .unwrap();
        assert_eq!(training_seen.get(), Some(ExecutionMode::Training));

        let inference_seen = Cell::new(None);
        Model::new(|cx: Cx| -> Result<Tensor> {
            inference_seen.set(Some(cx.mode()));
            cx.input(&[1])
        })
        .inference()
        .trace()
        .unwrap();
        assert_eq!(inference_seen.get(), Some(ExecutionMode::Inference));
    }

    fn classifier(cx: Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 4])?;
        let scope = cx.scope("head")?;
        let weight = scope.param("weight", &[4, 3])?;
        Ok(input.matmul(&weight)?)
    }

    #[test]
    fn trace_records_model_body_and_schema() {
        let (schema, init_output) = init(classifier).unwrap();
        assert_eq!(init_output.shape(), [2, 3]);
        assert_eq!(schema.inputs().len(), 1);
        assert_eq!(schema.inputs()[0].shape(), [2, 4]);
        assert_eq!(schema.inputs()[0].dtype(), DType::F32);
        assert_eq!(schema.parameters().len(), 1);
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.parameters()[0].shape(), [4, 3]);
        assert_eq!(
            schema.arguments(),
            &[ModelArgument::Input(0), ModelArgument::Parameter(0)]
        );

        let applied = apply(&schema, classifier).unwrap();
        assert_eq!(applied.outputs()[0].shape(), [2, 3]);
        assert_eq!(applied.prepare().unwrap().input_count(), 2);
    }

    #[test]
    fn parameter_schema_separates_checkpoint_and_compute_dtypes() {
        let (schema, value) =
            init(|cx| cx.param_with_dtypes("weight", &[2, 3], DType::F16, DType::F32)).unwrap();

        let parameter = &schema.parameters()[0];
        assert_eq!(parameter.storage_dtype(), DType::F16);
        assert_eq!(parameter.compute_dtype(), DType::F32);
        assert_eq!(value.dtype(), DType::F32);
        assert_eq!(value.shape(), [2, 3]);

        let error = match init(|cx| cx.param_with_dtypes("weight", &[2], DType::I32, DType::F32)) {
            Ok(_) => panic!("unsupported storage dtype was accepted"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            Error::UnsupportedParameterDType { dtype: DType::I32 }
        ));
    }

    #[test]
    fn ordinary_model_trace_invokes_the_apply_body_once() {
        let calls = Cell::new(0);
        let definition = Model::new(|cx: Cx, input: Tensor| {
            calls.set(calls.get() + 1);
            input.apply(&cx.named_layer("head", Linear::new(3))?)
        })
        .inputs(ModelInput::new([2, 4]));

        let applied = definition.trace().unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(applied.schema().parameters().len(), 2);
        assert_eq!(applied.outputs()[0].shape(), [2, 3]);
        assert_eq!(applied.prepare().unwrap().input_count(), 3);
    }

    #[test]
    fn all_resident_trace_invokes_the_apply_body_once() {
        let calls = Cell::new(0);
        let definition = Model::new(|cx: Cx, input: Tensor| {
            calls.set(calls.get() + 1);
            input.apply(&cx.named_layer("head", Linear::new(3))?)
        })
        .inputs(ModelInput::new([2, 4]));

        let (selection, applied) = definition.trace_resident_all().unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(selection.len(), 2);
        assert_eq!(applied.resident_parameters().count(), 2);
        assert_eq!(applied.prepare_stateful().unwrap().input_indices(), [0]);
    }

    #[test]
    fn schema_guided_resident_trace_is_single_pass_and_structural() {
        fn source(cx: Cx) -> Result<Tensor> {
            cx.param("weight", &[2])
        }

        let source = Model::new(source).trace().unwrap();
        let schema = source.schema().clone();
        let selection = schema.select_all();
        let calls = Cell::new(0);
        let compatible = Model::new(|cx: Cx| -> Result<_> {
            calls.set(calls.get() + 1);
            cx.param("weight", &[2])
        })
        .trace_resident(&selection)
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(compatible.resident_parameters().count(), 1);
        assert_eq!(compatible.parameter_tensors(&selection).unwrap().len(), 1);
        compatible.validate_resident_parameters(&selection).unwrap();

        let incompatible = Model::new(|cx: Cx| cx.param("weight", &[3])).trace_resident(&selection);
        assert!(matches!(incompatible, Err(Error::ModelSchemaMismatch)));
    }

    #[test]
    fn independent_scope_handles_preserve_parent_paths_and_validate_atomically() {
        let (schema, _) = init(|cx| {
            {
                let block = cx.scope_path(["encoder", "0"])?;
                block.param("weight", &[2])?;
            }
            assert!(cx.scope_path(["unused", "bad.segment"]).is_err());
            cx.param("root", &[1])
        })
        .unwrap();

        assert!(schema.get("encoder.0.weight").is_some());
        assert!(schema.get("root").is_some());
        assert!(schema.get("unused.root").is_none());
    }

    #[test]
    fn path_macro_builds_stable_mixed_segment_trees() {
        let (schema, _) = init(|cx| {
            let block = path!(cx / "encoder" / "blocks" / 17usize)?;
            assert_eq!(block.model_path().to_string(), "encoder.blocks.17");
            block.param("weight", &[2])?;
            cx.param("root", &[1])
        })
        .unwrap();

        assert!(schema.get("encoder.blocks.17.weight").is_some());
        assert!(schema.get("root").is_some());
    }

    #[test]
    fn model_conditional_hoists_branch_parameters_into_the_abi() {
        let applied = Model::new(|cx: Cx| -> Result<Tensor> {
            let predicate = cx.input_dtype(&[], DType::I32)?;
            cx.cond(
                &predicate,
                |cx| cx.scope("positive")?.param("weight", &[2]),
                |cx| cx.scope("negative")?.param("weight", &[2]),
            )
        })
        .trace()
        .unwrap();

        assert_eq!(applied.outputs()[0].shape(), [2]);
        assert!(applied.schema().get("positive.weight").is_some());
        assert!(applied.schema().get("negative.weight").is_some());
        assert_eq!(applied.prepare().unwrap().input_count(), 3);
    }

    #[test]
    fn model_conditional_merges_state_versions_as_hidden_results() {
        let applied = Model::new(|cx: Cx| -> Result<Tensor> {
            let state = cx.state("counter", &[], DType::F32)?;
            let value = state.read(&cx)?;
            let predicate = cx.input_dtype(&[], DType::I32)?;
            let visible = cx.cond(
                &predicate,
                |cx| {
                    let next = value.add_scalar(1.0)?;
                    state.write(&cx, &next)?;
                    Ok(next)
                },
                |cx| {
                    let next = value.add_scalar(-1.0)?;
                    state.write(&cx, &next)?;
                    Ok(next)
                },
            )?;
            Ok(visible.add(&state.read(&cx)?)?)
        })
        .trace()
        .unwrap();

        let prepared = applied.prepare_stateful().unwrap();
        let mlir = std::str::from_utf8(prepared.lowered_program().code()).unwrap();
        assert!(mlir.contains("stablehlo.if"));
        assert!(mlir.contains("tensor<f32>, tensor<f32>"));
    }

    #[test]
    fn byte_inputs_can_be_preprocessed_inside_the_model_program() {
        fn preprocess(cx: Cx) -> Result<Tensor> {
            Ok(cx
                .input_dtype(&[2, 2, 3], DType::U8)?
                .cast(DType::F32)?
                .mul_scalar(1.0 / 255.0)?)
        }

        let (schema, _) = init(preprocess).unwrap();
        assert_eq!(schema.inputs()[0].dtype(), DType::U8);
        let applied = apply(&schema, preprocess).unwrap();
        let lowered = applied.prepare().unwrap();
        assert_eq!(lowered.input_spec(0).unwrap().dtype, DType::U8);
        assert_eq!(lowered.output_spec(0).unwrap().dtype, DType::F32);
        let stablehlo = std::str::from_utf8(lowered.code()).unwrap();
        assert!(stablehlo.contains("stablehlo.convert"));
    }

    #[test]
    fn selected_parameter_tensors_drive_partial_autodiff() {
        fn product(cx: Cx) -> Result<Tensor> {
            let body = cx.scope("body")?.param("weight", &[2])?;
            let head = cx.scope("head")?.param("weight", &[2])?;
            Ok(body.mul(&head)?.sum(&[0], false)?)
        }

        let (schema, _) = init(product).unwrap();
        let applied = apply(&schema, product).unwrap();
        let head = applied.schema().select_under("head");
        let leaves = applied.parameter_tensors(&head).unwrap();
        let gradients = applied.outputs()[0].grad(&leaves).unwrap();

        assert_eq!(leaves.len(), 1);
        assert_eq!(gradients.len(), 1);
        assert_eq!(gradients[0].shape(), [2]);
    }

    #[test]
    fn selected_parameter_tensors_reject_another_schema() {
        let (schema, _) = init(|cx| cx.param("weight", &[2])).unwrap();
        let applied = apply(&schema, |cx| cx.param("weight", &[2])).unwrap();
        let (other, _) = init(|cx| cx.param("weight", &[3])).unwrap();
        let error = match applied.parameter_tensors(&other.select_all()) {
            Ok(_) => panic!("a structurally different schema was accepted"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::SelectionSchemaMismatch));

        let (same_shape, _) = init(|cx| cx.param("weight", &[2])).unwrap();
        assert!(matches!(
            applied.parameter_tensors(&same_shape.select_all()),
            Err(Error::SelectionSchemaMismatch)
        ));
    }

    #[test]
    fn repeated_parameter_reads_share_one_input() {
        let (schema, _) = init(|cx| {
            let first = cx.param("weight", &[2, 2])?;
            let second = cx.param("weight", &[2, 2])?;
            Ok(first.add(&second)?)
        })
        .unwrap();
        let applied = apply(&schema, |cx| {
            let first = cx.param("weight", &[2, 2])?;
            let second = cx.param("weight", &[2, 2])?;
            Ok(vec![first.add(&second)?])
        })
        .unwrap();
        assert_eq!(applied.prepare().unwrap().input_count(), 1);
    }

    #[test]
    fn bind_rejects_wrong_number_of_model_inputs_before_buffer_access() {
        let (schema, _) = init(classifier).unwrap();
        let applied = apply(&schema, classifier).unwrap();
        let no_inputs: [&Buffer; 0] = [];
        let error = match applied.bind(no_inputs, std::iter::empty::<(&str, &Buffer)>()) {
            Ok(_) => panic!("missing model input unexpectedly bound"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            Error::InputCount {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn linear_infers_input_features_at_its_use_site() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 4])?;
            input.apply(&cx.named_layer("head", Linear::new(3))?)
        })
        .unwrap();
        assert_eq!(output.shape(), [2, 3]);
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.parameters()[0].shape(), [3, 4]);
        assert_eq!(
            schema.parameters()[0].initializer(),
            Some(Initializer::kaiming_uniform())
        );
        assert_eq!(schema.parameters()[1].path(), "head.bias");
        assert_eq!(schema.parameters()[1].shape(), [3]);
        assert!(matches!(
            schema.parameters()[1].initializer(),
            Some(initializer) if initializer == Initializer::uniform(-0.5, 0.5)
        ));
    }

    #[test]
    fn program_owns_the_model_body_and_layer_builders_own_scopes() {
        let model = Model::new(|cx: Cx| -> Result<_> {
            let input = cx.input(&[2, 4])?;
            let hidden = cx
                .scope("hidden")?
                .linear(8)
                .bias(false)
                .apply(&input)?
                .relu()?;
            hidden.apply(&cx.named_layer("head", Linear::new(3))?)
        });

        let applied = model.trace().unwrap();
        let schema = applied.schema();

        assert_eq!(schema.parameters()[0].path(), "hidden.weight");
        assert_eq!(schema.parameters()[0].shape(), [8, 4]);
        assert_eq!(schema.parameters()[1].path(), "head.weight");
        assert_eq!(schema.parameters()[1].shape(), [3, 8]);
        assert_eq!(schema.parameters()[2].path(), "head.bias");
        assert_eq!(applied.outputs()[0].shape(), [2, 3]);
    }

    #[test]
    fn one_context_composes_parameters_and_resident_state() {
        let model = Model::new(|cx: Cx| -> Result<_> {
            let input = cx.input(&[2, 4])?;
            let output = input.apply(&cx.named_layer("head", Linear::new(3))?)?;
            let count = cx.state("steps", &[], DType::I32)?;
            let next = count.read(&cx)?.wrapping_add_scalar(1)?;
            count.write(&cx, &next)?;
            let noise = cx.rng("sampling")?.normal_f32(output.shape())?;
            Ok(output.add(&noise.mul_scalar(0.0)?)?)
        });

        let applied = model.trace().unwrap();
        let schema = applied.schema();
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.states()[0].path(), "steps");
        assert_eq!(schema.states()[1].path(), "sampling.key0");
        assert!(applied.is_stateful());
        assert!(matches!(
            applied.prepare(),
            Err(Error::StatefulOperation {
                operation: "prepare"
            })
        ));
        let prepared = applied.prepare_stateful().unwrap();
        let (_, steps) = applied.states().next().unwrap();
        assert_eq!(prepared.state_type(steps).unwrap(), (DType::I32, vec![]));
    }

    #[test]
    fn invalid_random_probabilities_are_typed_errors() {
        let result = Model::new(|cx: Cx| {
            let input = cx.input(&[2])?;
            Ok(cx.rng("dropout")?.dropout(&input, f32::NAN)?.output)
        })
        .trace();

        assert!(matches!(
            result,
            Err(Error::InvalidProbability {
                operation: "dropout keep",
                probability,
                ..
            }) if probability.is_nan()
        ));
    }

    #[test]
    fn transforms_append_named_resident_state() {
        let mut applied = Model::new(|cx: Cx| cx.input(&[2])).trace().unwrap();
        let moment = applied
            .transform_state("__transform.moment", &[2], DType::F32)
            .unwrap();
        let next = moment.value().add_scalar(1.0).unwrap();
        applied.write_transform_states(&[(&moment, &next)]).unwrap();

        assert_eq!(moment.path(), "__transform.moment");
        assert!(matches!(
            applied.transform_state("__transform.moment", &[2], DType::F32),
            Err(Error::DuplicateTransformState { .. })
        ));
        let prepared = applied.prepare_stateful().unwrap();
        let (_, slot) = applied.states().next().unwrap();
        assert_eq!(prepared.state_type(slot).unwrap(), (DType::F32, vec![2]));
    }

    #[test]
    fn selected_parameters_can_be_traced_as_resident_state() {
        let calls = Cell::new(0);
        let definition = Model::new(|cx: Cx, input: Tensor| {
            calls.set(calls.get() + 1);
            let body = input.apply(&cx.named_layer("body", Linear::new(3).bias(false))?)?;
            body.apply(&cx.named_layer("head", Linear::new(2).bias(false))?)
        })
        .inputs(ModelInput::new([1, 3]));
        let (selection, applied) = definition.trace_resident_under("head").unwrap();
        let schema = applied.schema();

        assert_eq!(calls.get(), 1);
        assert!(applied.is_stateful());
        assert_eq!(selection.len(), 1);
        assert_eq!(schema.parameters().len(), 2);
        assert_eq!(applied.resident_parameters().count(), 1);
        let prepared = applied.prepare_stateful().unwrap();
        assert_eq!(prepared.input_indices(), [0, 1]);
        let (_, spec, slot) = applied.resident_parameters().next().unwrap();
        assert_eq!(spec.path(), "head.weight");
        assert_eq!(prepared.state_type(slot).unwrap(), (DType::F32, vec![2, 3]));

        let quantized = Model::new(|cx: Cx| -> Result<_> {
            Ok(cx
                .param_dtype("weight", &[2], DType::U8)?
                .cast(DType::F32)?)
        });
        let (_, applied) = quantized.trace_resident_all().unwrap();
        let prepared = applied.prepare_stateful().unwrap();
        let (_, _, slot) = applied.resident_parameters().next().unwrap();
        assert_eq!(prepared.state_type(slot).unwrap(), (DType::U8, vec![2]));
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn resident_parameter_executes_without_a_parameter_argument() {
        let client = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .unwrap();
        let definition = Model::new(|cx: Cx, input: Tensor| {
            input.apply(&cx.named_layer("head", Linear::new(2).bias(false))?)
        })
        .inputs(ModelInput::new([1, 3]));
        let (_, applied) = definition.trace_resident_all().unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let program = applied.compile_stateful(&mut compiler).unwrap();
        assert!(matches!(
            program.session().build(),
            Err(Error::MissingResidentParameterInitializer { .. })
        ));
        let wrong_shape = client.buffer(&[3, 2], &[0.0; 6]).unwrap();
        assert!(matches!(
            program.session().parameter("head.weight", wrong_shape),
            Err(Error::BufferShape { .. })
        ));
        let weight = client
            .buffer(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .unwrap();
        let mut session = program
            .session()
            .parameters([("head.weight", weight)])
            .unwrap()
            .build()
            .unwrap();
        let input = client.buffer(&[1, 3], &[2.0, 3.0, 4.0]).unwrap();
        let output: Buffer = session.run(&input).unwrap();
        assert_eq!(output.to_vec::<f32>().unwrap(), [2.0, 3.0]);
        let parameters = applied.resident_parameter_buffers(session.raw()).unwrap();
        assert_eq!(parameters.len(), 1);
        assert_eq!(parameters[0].0, "head.weight");
        assert_eq!(
            parameters[0].1.to_vec::<f32>().unwrap(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );

        applied.take_session(session.into_raw()).unwrap();
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn unified_context_state_and_rng_execute_as_one_program() {
        let model = Model::new(|cx: Cx| -> Result<_> {
            let steps = cx.state("steps", &[], DType::I32)?;
            let next = steps.read(&cx)?.wrapping_add_scalar(1)?;
            steps.write(&cx, &next)?;
            let draw = cx.rng("sampling")?.uniform_f32(&[4])?;
            Ok([draw, next])
        });
        let applied = model.trace().unwrap();
        let client =
            unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").expect("PJRT plugin path")) }
                .unwrap();
        let mut compiler = Compiler::new(client, CacheLimits::default());
        let program = applied.compile_stateful(&mut compiler).unwrap();
        let mut session = program
            .session()
            .rng_seed("sampling", 42)
            .unwrap()
            .build()
            .unwrap();

        let first: [Buffer; 2] = session.run(&[] as &[&Buffer]).unwrap();
        let second: [Buffer; 2] = session.run(&[] as &[&Buffer]).unwrap();
        assert_ne!(
            first[0].to_vec::<f32>().unwrap(),
            second[0].to_vec::<f32>().unwrap()
        );
        assert_eq!(first[1].to_vec::<i32>().unwrap(), [1]);
        assert_eq!(second[1].to_vec::<i32>().unwrap(), [2]);
    }

    #[test]
    fn conv2d_infers_checkpoint_kernel_input_channels() {
        let options = Conv2dOptions {
            padding: [[1, 1], [1, 1]],
            groups: 2,
            ..Default::default()
        };
        let model = |cx: Cx| {
            let input = cx.input(&[1, 8, 8, 4])?;
            cx.scope("conv_in")?
                .conv2d(6, [3, 3])
                .options(options)
                .apply(&input)
        };
        let (schema, output) = init(model).unwrap();
        assert_eq!(output.shape(), [1, 8, 8, 6]);
        assert_eq!(schema.parameters()[0].path(), "conv_in.weight");
        assert_eq!(schema.parameters()[0].shape(), [6, 2, 3, 3]);
        assert_eq!(schema.parameters()[1].path(), "conv_in.bias");
        assert_eq!(schema.parameters()[1].shape(), [6]);

        let applied = apply(&schema, model).unwrap();
        let lowered = applied.prepare().unwrap();
        assert!(
            std::str::from_utf8(lowered.code())
                .unwrap()
                .contains("stablehlo.convolution")
        );
    }

    #[test]
    fn group_norm_nhwc_infers_affine_channel_shape() {
        let model = |cx: Cx| {
            let input = cx.input(&[1, 8, 8, 32])?;
            input.apply(&cx.named_layer("norm", GroupNorm::new(8))?)
        };
        let (schema, output) = init(model).unwrap();
        assert_eq!(output.shape(), [1, 8, 8, 32]);
        assert_eq!(schema.parameters()[0].path(), "norm.weight");
        assert_eq!(schema.parameters()[0].shape(), [32]);
        assert_eq!(schema.parameters()[1].path(), "norm.bias");
        assert_eq!(schema.parameters()[1].shape(), [32]);
        apply(&schema, model).unwrap().prepare().unwrap();
    }

    #[test]
    fn layer_norm_infers_trailing_affine_shape() {
        let (schema, output) = init(|cx| {
            let input = cx.input(&[2, 7, 32])?;
            input.apply(&cx.named_layer("norm", LayerNorm::new(1))?)
        })
        .unwrap();
        assert_eq!(output.shape(), [2, 7, 32]);
        assert_eq!(schema.parameters()[0].shape(), [32]);
        assert_eq!(schema.parameters()[1].shape(), [32]);
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH and PJRT_CUDA_PLUGIN_PATH"]
    fn parameter_effect_model_executes_on_cpu_and_cuda() {
        let cpu = unsafe {
            Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH").expect("CPU plugin path"))
        }
        .expect("load CPU plugin");
        let cuda_options = ClientOptions::new().set("preallocate", false);
        let cuda = unsafe {
            Client::load_with_options(
                std::env::var("PJRT_CUDA_PLUGIN_PATH").expect("CUDA plugin path"),
                &cuda_options,
            )
        }
        .expect("load CUDA plugin");

        let model = |cx: Cx| {
            let input = cx.input(&[2, 3])?;
            Ok(cx
                .scope("head")?
                .linear(2)
                .bias(false)
                .apply(&input)?
                .relu()?)
        };
        let (schema, _) = init(model).expect("initialize model schema");
        let applied = apply(&schema, model).expect("trace model application");

        for client in [cpu, cuda] {
            let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
            let compiled = applied.compile(&mut compiler).expect("compile model");
            let input = client
                .buffer(&[2, 3], &[1., -2., 3., -4., 5., -6.])
                .expect("upload input");
            let weight = client
                .buffer(&[2, 3], &[1., 3., 5., 2., 4., 6.])
                .expect("upload parameter");
            let output: Buffer = compiled
                .run([&input], [("head.weight", &weight)])
                .expect("execute model");
            assert_eq!(output.to_vec::<f32>().unwrap(), [10., 12., 0., 0.]);
        }
    }
}
