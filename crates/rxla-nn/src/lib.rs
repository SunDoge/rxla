//! Scoped parameter effects for tensor tracing.
//!
//! A [`Cx`] is deliberately not a global variable store. `init` and `apply`
//! interpret the same `param` calls differently, so model code is written once
//! without making parameter identity depend on call order.

use rxla_core::{DType, StateGraph, StateSlot, Tensor};
use snafu::{OptionExt, Snafu, ensure};
use std::collections::{BTreeMap, HashSet};
use std::ops::{Deref, DerefMut};

mod applied;
pub use applied::{
    AppliedModel, BoundModel, CompiledModel, CompiledModelSessionBuilder, CompiledStatefulModel,
    ModelArguments, ModelSession, ModelSessionBuffers, ModelSessionBuilder, TransformState,
};
mod inputs;
pub use inputs::{ModelHandler, ModelInput, ModelInputValues, ModelInputs};
mod layers;
pub use layers::{
    Conv2d, Embedding, GroupNorm, Layer, LayerNorm, Linear, QuantizedLinear, RmsNorm,
};
mod outputs;
pub use outputs::{ModelOutputValues, ModelOutputs};
mod schema;
pub use schema::{ModelArgument, ModelInputSpec, ParamSchema, ParameterSpec, StateSpec};
mod selection;
pub use selection::{ParameterId, ParameterSelection};

/// A reusable effect-based model definition.
///
/// It captures one ordinary Rust function so callers do not repeat closures
/// around schema discovery and application. Compiled execution artifacts use
/// [`rxla_core::Program`]; keeping the names distinct avoids import aliases in
/// applications that construct and run models in the same module.
pub struct Model<F, I = NoModelInputs> {
    apply: F,
    inputs: I,
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
    #[snafu(display("model effect at index {index} is incompatible with the schema"))]
    EffectMismatch { index: usize },
    #[snafu(display("model input {index} is absent from the schema"))]
    UnexpectedInput { index: usize },
    #[snafu(display("model input declaration at index {index} is incompatible with the schema"))]
    IncompatibleInput { index: usize },
    #[snafu(display("apply did not read schema parameter {path:?}"))]
    UnreadParameter { path: String },
    #[snafu(display("apply stopped before schema input {index}"))]
    UnreadInput { index: usize },
    #[snafu(display("apply stopped before schema effect {index}"))]
    UnreadEffect { index: usize },
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
    #[snafu(display("invalid model definition: {message}"))]
    InvalidDefinition { message: String },
    #[snafu(display("state declaration for {path:?} is incompatible with the schema"))]
    IncompatibleState { path: String },
    #[snafu(display("transform state {path:?} is already declared"))]
    DuplicateTransformState { path: String },
    #[snafu(display("model returned {actual} output buffers, expected {expected}"))]
    OutputCount { expected: usize, actual: usize },
    #[snafu(display("output buffers do not match the requested result structure"))]
    OutputStructure,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl<F> Model<F, NoModelInputs> {
    /// Capture one model `apply` function for repeated effect interpretation.
    pub fn new(apply: F) -> Self {
        Self {
            apply,
            inputs: NoModelInputs,
        }
    }

    /// Supply structured input declarations to an `apply(cx, inputs)` function.
    pub fn inputs<I>(self, inputs: I) -> Model<F, I> {
        Model {
            apply: self.apply,
            inputs,
        }
    }
}

impl<F, T> Model<F, NoModelInputs>
where
    F: Fn(&mut Cx) -> Result<T>,
    T: ModelOutputs,
{
    /// Discover parameter effects and produce the executable model trace.
    ///
    /// This is the ordinary one-call path. [`Self::init`] and [`Self::apply`]
    /// remain available when callers need to inspect or restore a schema
    /// between the two interpretations.
    pub fn trace(&self) -> Result<(ParamSchema, AppliedModel)> {
        let schema = self.init()?;
        let applied = self.apply(&schema)?;
        Ok((schema, applied))
    }

    /// Discover the schema and trace a caller-selected resident parameter set.
    ///
    /// The owned selection is returned for later transforms such as SGD or
    /// Adam; it retains schema identity without borrowing the returned schema.
    pub fn trace_resident(
        &self,
        select: impl FnOnce(&ParamSchema) -> ParameterSelection,
    ) -> Result<(ParamSchema, ParameterSelection, AppliedModel)> {
        let schema = self.init()?;
        let selection = select(&schema);
        let applied = self.apply_resident(&schema, &selection)?;
        Ok((schema, selection, applied))
    }

    /// Discover the input/parameter effect schema from this model body.
    pub fn init(&self) -> Result<ParamSchema> {
        init(|cx| (self.apply)(cx)).map(|(schema, _)| schema)
    }

    /// Trace this model body against a previously discovered schema.
    pub fn apply(&self, schema: &ParamSchema) -> Result<AppliedModel> {
        apply(schema, |cx| (self.apply)(cx))
    }

    /// Trace with selected parameters stored as resident session state.
    pub fn apply_resident(
        &self,
        schema: &ParamSchema,
        selection: &ParameterSelection,
    ) -> Result<AppliedModel> {
        apply_resident(schema, selection, |cx| (self.apply)(cx))
    }
}

impl<F, I> Model<F, I>
where
    I: ModelInputs,
{
    /// Discover the schema and trace `apply` with structured lazy inputs.
    pub fn trace<Marker>(&self) -> Result<(ParamSchema, AppliedModel)>
    where
        F: ModelHandler<I, Marker>,
    {
        let schema = self.init()?;
        let applied = self.apply(&schema)?;
        Ok((schema, applied))
    }

    /// Discover typed inputs and trace a caller-selected resident parameter set.
    pub fn trace_resident<Marker>(
        &self,
        select: impl FnOnce(&ParamSchema) -> ParameterSelection,
    ) -> Result<(ParamSchema, ParameterSelection, AppliedModel)>
    where
        F: ModelHandler<I, Marker>,
    {
        let schema = self.init()?;
        let selection = select(&schema);
        let applied = self.apply_resident(&schema, &selection)?;
        Ok((schema, selection, applied))
    }

    pub fn init<Marker>(&self) -> Result<ParamSchema>
    where
        F: ModelHandler<I, Marker>,
    {
        init(|cx| self.apply.invoke(cx, &self.inputs)).map(|(schema, _)| schema)
    }

    pub fn apply<Marker>(&self, schema: &ParamSchema) -> Result<AppliedModel>
    where
        F: ModelHandler<I, Marker>,
    {
        apply(schema, |cx| self.apply.invoke(cx, &self.inputs))
    }

    /// Trace with selected parameters stored as resident session state.
    pub fn apply_resident<Marker>(
        &self,
        schema: &ParamSchema,
        selection: &ParameterSelection,
    ) -> Result<AppliedModel>
    where
        F: ModelHandler<I, Marker>,
    {
        apply_resident(schema, selection, |cx| self.apply.invoke(cx, &self.inputs))
    }
}

enum ParamMode {
    Init {
        schema: ParamSchema,
        values: BTreeMap<String, Tensor>,
    },
    Apply {
        schema: ParamSchema,
        values: BTreeMap<String, Tensor>,
        read: HashSet<String>,
        resident: HashSet<ParameterId>,
    },
}

/// The explicit interpreter for scoped parameter effects.
///
/// Model functions receive this same type during [`init`] and [`apply`]. The
/// mode is selected by the caller, never inferred from prior invocations.
pub struct Cx {
    graph: StateGraph,
    scope: Vec<String>,
    input_index: usize,
    effect_index: usize,
    mode: ParamMode,
    states: BTreeMap<String, StateDeclaration>,
    rngs: BTreeMap<String, RngStream>,
    resident_parameters: BTreeMap<String, StateSlot>,
}

/// A temporary lexical effect scope.
///
/// It dereferences to [`Cx`] but adds no layer methods of its own, so nested
/// model code can use any built-in or third-party builder without name
/// collisions. Dropping the guard restores the parent path.
pub struct Scope<'a> {
    cx: &'a mut Cx,
    parent_depth: usize,
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        self.cx.scope.truncate(self.parent_depth);
    }
}

impl Deref for Scope<'_> {
    type Target = Cx;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl DerefMut for Scope<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.cx
    }
}

struct StateDeclaration {
    slot: StateSlot,
    shape: Vec<i64>,
    dtype: DType,
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
pub struct Rng<'a> {
    stream: &'a mut RngStream,
}

impl Cx {
    fn init() -> Self {
        Self {
            graph: StateGraph::default(),
            scope: Vec::new(),
            input_index: 0,
            effect_index: 0,
            mode: ParamMode::Init {
                schema: ParamSchema::default(),
                values: BTreeMap::new(),
            },
            states: BTreeMap::new(),
            rngs: BTreeMap::new(),
            resident_parameters: BTreeMap::new(),
        }
    }

    fn apply(schema: ParamSchema, resident: HashSet<ParameterId>) -> Self {
        Self {
            graph: StateGraph::default(),
            scope: Vec::new(),
            input_index: 0,
            effect_index: 0,
            mode: ParamMode::Apply {
                schema,
                values: BTreeMap::new(),
                read: HashSet::new(),
                resident,
            },
            states: BTreeMap::new(),
            rngs: BTreeMap::new(),
            resident_parameters: BTreeMap::new(),
        }
    }

    /// Declare/read an F32 parameter at the current lexical scope.
    pub fn param(&mut self, name: &str, shape: &[i64]) -> Result<Tensor> {
        self.param_dtype(name, shape, DType::F32)
    }

    /// Declare/read a parameter with its storage dtype.
    ///
    /// The current symbolic tensor surface supports F32 parameters, frozen
    /// BF16 storage exposed as F32 computation values, and raw U8 storage for
    /// explicitly dequantized inference layers. Other dtypes are rejected.
    pub fn param_dtype(&mut self, name: &str, shape: &[i64], dtype: DType) -> Result<Tensor> {
        validate_name(name)?;
        let path = self.path(name);
        ensure!(
            shape.iter().all(|&dim| dim >= 0),
            NegativeParameterDimensionSnafu { path }
        );
        let requested = ParameterSpec {
            path: path.clone(),
            shape: shape.to_vec(),
            dtype,
        };
        let (graph, mode, resident_parameters) = (
            &mut self.graph,
            &mut self.mode,
            &mut self.resident_parameters,
        );
        match mode {
            ParamMode::Init { schema, values } => {
                if let Some(existing) = schema.get(&path) {
                    ensure!(existing == &requested, IncompatibleParameterSnafu { path });
                    return Ok(values
                        .get(&path)
                        .expect("schema and parameter value are inserted together")
                        .clone());
                }
                let value = parameter_tensor(graph, shape, dtype)?;
                schema.push_parameter(requested);
                self.effect_index += 1;
                values.insert(path, value.clone());
                Ok(value)
            }
            ParamMode::Apply {
                schema,
                values,
                read,
                resident,
            } => {
                let expected = schema
                    .get(&path)
                    .with_context(|| UnknownParameterSnafu { path: path.clone() })?;
                ensure!(expected == &requested, IncompatibleParameterSnafu { path });
                read.insert(path.clone());
                if let Some(value) = values.get(&path) {
                    return Ok(value.clone());
                }
                let parameter_index = schema
                    .parameter_id(&path)
                    .expect("every parameter schema entry has an index")
                    .index();
                ensure!(
                    schema.arguments().get(self.effect_index)
                        == Some(&ModelArgument::Parameter(parameter_index)),
                    EffectMismatchSnafu {
                        index: self.effect_index
                    }
                );
                let value = if resident.contains(&ParameterId::from_index(parameter_index)) {
                    let slot = graph.state_named(&format!("__parameter.{path}"), shape, dtype)?;
                    let stored = graph.read(&slot)?;
                    let value = match dtype {
                        DType::BF16 => stored.cast(DType::F32)?,
                        DType::F32 | DType::U8 => stored,
                        _ => return UnsupportedParameterDTypeSnafu { dtype }.fail(),
                    };
                    resident_parameters.insert(path.clone(), slot);
                    value
                } else {
                    parameter_tensor(graph, shape, dtype)?
                };
                values.insert(path, value.clone());
                self.effect_index += 1;
                Ok(value)
            }
        }
    }

    /// Enter a lexical parameter/effect scope without a closure.
    pub fn scope(&mut self, name: &str) -> Result<Scope<'_>> {
        self.scope_path([name])
    }

    /// Enter several lexical path segments with one RAII guard.
    pub fn scope_path<I, S>(&mut self, segments: I) -> Result<Scope<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let segments = segments
            .into_iter()
            .map(|segment| segment.as_ref().to_owned())
            .collect::<Vec<_>>();
        for segment in &segments {
            validate_name(segment)?;
        }
        let parent_depth = self.scope.len();
        self.scope.extend(segments);
        Ok(Scope {
            cx: self,
            parent_depth,
        })
    }

    /// Create a visible F32 model input in this trace.
    pub fn input(&mut self, shape: &[i64]) -> Result<Tensor> {
        self.input_dtype(shape, DType::F32)
    }

    /// Create a visible input with an explicit dtype.
    pub fn input_dtype(&mut self, shape: &[i64], dtype: DType) -> Result<Tensor> {
        let requested = ModelInputSpec {
            shape: shape.to_vec(),
            dtype,
        };
        let input_index = self.input_index;
        match &mut self.mode {
            ParamMode::Init { schema, .. } => {
                let declared = schema.push_input(requested.clone());
                debug_assert_eq!(declared, input_index);
            }
            ParamMode::Apply { schema, .. } => {
                let expected = schema
                    .inputs()
                    .get(input_index)
                    .context(UnexpectedInputSnafu { index: input_index })?;
                ensure!(
                    expected == &requested,
                    IncompatibleInputSnafu { index: input_index }
                );
                ensure!(
                    schema.arguments().get(self.effect_index)
                        == Some(&ModelArgument::Input(input_index)),
                    EffectMismatchSnafu {
                        index: self.effect_index
                    }
                );
            }
        }
        self.input_index += 1;
        self.effect_index += 1;
        match dtype {
            DType::F32 | DType::I32 | DType::U8 => Ok(self.graph.input_with_dtype(shape, dtype)?),
            DType::BF16 => Ok(self.graph.input_bf16_as_f32(shape)?),
            _ => UnsupportedInputDTypeSnafu { dtype }.fail(),
        }
    }

    /// Create a graph-local I32 coordinate tensor without adding an ABI input.
    pub fn iota_i32(&self, shape: &[i64], axis: usize) -> Result<Tensor> {
        Ok(self.graph.iota_i32(shape, axis)?)
    }

    /// Create a graph-local F32 constant without adding an ABI input.
    pub fn constant(&self, shape: &[i64], values: &[f32]) -> Result<Tensor> {
        Ok(self.graph.constant(shape, values)?)
    }

    /// Declare or read named resident state at the current lexical scope.
    pub fn state(&mut self, name: &str, shape: &[i64], dtype: DType) -> Result<State> {
        validate_name(name)?;
        let path = self.path(name);
        if let Some(existing) = self.states.get(&path) {
            ensure!(
                existing.shape == shape && existing.dtype == dtype,
                IncompatibleStateSnafu { path }
            );
            return Ok(State {
                path,
                slot: existing.slot.clone(),
            });
        }
        let state_index = match &mut self.mode {
            ParamMode::Init { schema, .. } => schema.push_state(StateSpec {
                path: path.clone(),
                shape: shape.to_vec(),
                dtype,
            }),
            ParamMode::Apply { schema, .. } => {
                let (index, expected) = schema
                    .states()
                    .iter()
                    .enumerate()
                    .find(|(_, state)| state.path == path)
                    .context(IncompatibleStateSnafu { path: path.clone() })?;
                ensure!(
                    expected.shape == shape && expected.dtype == dtype,
                    IncompatibleStateSnafu { path }
                );
                ensure!(
                    schema.arguments().get(self.effect_index) == Some(&ModelArgument::State(index)),
                    EffectMismatchSnafu {
                        index: self.effect_index
                    }
                );
                index
            }
        };
        let _ = state_index;
        let slot = self.graph.state_named(&path, shape, dtype)?;
        self.effect_index += 1;
        self.states.insert(
            path.clone(),
            StateDeclaration {
                slot: slot.clone(),
                shape: shape.to_vec(),
                dtype,
            },
        );
        Ok(State { path, slot })
    }

    /// Borrow a named Threefry stream. Repeated calls continue the same stream.
    pub fn rng(&mut self, name: &str) -> Result<Rng<'_>> {
        validate_name(name)?;
        let path = self.path(name);
        if !self.rngs.contains_key(&path) {
            let words = {
                let mut scope = self.scope(name)?;
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
            self.rngs.insert(
                path.clone(),
                RngStream {
                    key: [values[0].clone(), values[1].clone()],
                    counters: [words[2].clone(), words[3].clone()],
                    next: [values[2].clone(), values[3].clone()],
                    available: self.graph.constant(&[], &[1.0])?,
                },
            );
        }
        Ok(Rng {
            stream: self.rngs.get_mut(&path).expect("inserted above"),
        })
    }

    fn validate_state(&self, state: &State) -> Result<()> {
        match self.states.get(&state.path) {
            Some(declaration) if declaration.slot.identity() == state.slot.identity() => Ok(()),
            _ => InvalidDefinitionSnafu {
                message: "state belongs to another model trace",
            }
            .fail(),
        }
    }

    fn finish_rngs(&mut self) -> Result<()> {
        for stream in std::mem::take(&mut self.rngs).into_values() {
            self.graph.write_many_if(
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
        if self.scope.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{}", self.scope.join("."), name)
        }
    }

    fn finish_apply(&self) -> Result<()> {
        let ParamMode::Apply { schema, read, .. } = &self.mode else {
            return Ok(());
        };
        if let Some(missing) = schema
            .parameters()
            .iter()
            .find(|parameter| !read.contains(parameter.path()))
        {
            return UnreadParameterSnafu {
                path: missing.path(),
            }
            .fail();
        }
        ensure!(
            self.input_index == schema.inputs().len(),
            UnreadInputSnafu {
                index: self.input_index,
            }
        );
        ensure!(
            self.effect_index == schema.arguments().len(),
            UnreadEffectSnafu {
                index: self.effect_index,
            }
        );
        Ok(())
    }

    fn parameter_tensors(&self) -> Vec<Tensor> {
        let ParamMode::Apply { schema, values, .. } = &self.mode else {
            unreachable!("only apply contexts expose parameter tensors")
        };
        schema
            .parameters()
            .iter()
            .map(|parameter| {
                values
                    .get(parameter.path())
                    .expect("finish_apply verified every schema parameter")
                    .clone()
            })
            .collect()
    }

    fn state_slots(&self, schema: &ParamSchema) -> Vec<(String, StateSlot)> {
        schema
            .states()
            .iter()
            .map(|state| {
                (
                    state.path().to_owned(),
                    self.states[state.path()].slot.clone(),
                )
            })
            .collect()
    }

    fn resident_parameter_slots(&self, schema: &ParamSchema) -> Vec<Option<StateSlot>> {
        schema
            .parameters()
            .iter()
            .map(|parameter| self.resident_parameters.get(parameter.path()).cloned())
            .collect()
    }

    fn into_schema(self) -> ParamSchema {
        match self.mode {
            ParamMode::Init { schema, .. } => schema,
            ParamMode::Apply { .. } => unreachable!("only init contexts produce schemas"),
        }
    }
}

impl State {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn read(&self, cx: &Cx) -> Result<Tensor> {
        cx.validate_state(self)?;
        Ok(cx.graph.read(&self.slot)?)
    }

    pub fn write(&self, cx: &mut Cx, value: &Tensor) -> Result<()> {
        cx.validate_state(self)?;
        Ok(cx.graph.write(&self.slot, value)?)
    }

    pub fn add_(&self, cx: &mut Cx, value: &Tensor) -> Result<()> {
        let next = self.read(cx)?.add(value)?;
        self.write(cx, &next)
    }
}

impl Rng<'_> {
    fn draw(&mut self, shape: &[i64]) -> Result<rxla_core::random::ThreefryBlocks> {
        let draw = rxla_core::random::threefry2x32_blocks(
            [&self.stream.key[0], &self.stream.key[1]],
            [&self.stream.next[0], &self.stream.next[1]],
            shape,
        )?;
        self.stream.available = self
            .stream
            .available
            .mul(&draw.counter_wrapped.neg()?.add_scalar(1.0)?)?;
        self.stream.next = draw.next_counter.clone();
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
            InvalidDefinitionSnafu {
                message: "device Bernoulli probability must be finite and in [0, 1]"
            }
        );
        if probability == 0.0 || probability == 1.0 {
            return Ok(self.stream.key[0]
                .scalar(probability)?
                .broadcast_to(shape)?);
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
            InvalidDefinitionSnafu {
                message: "dropout keep probability must be finite and in (0, 1]"
            }
        );
        let keep_mask = self.bernoulli(input.shape(), keep_probability)?;
        let output = input.dropout_with_mask(&keep_mask, keep_probability)?;
        Ok(rxla_core::random::DropoutSample { output, keep_mask })
    }
}

fn parameter_tensor(graph: &mut StateGraph, shape: &[i64], dtype: DType) -> Result<Tensor> {
    match dtype {
        DType::F32 => Ok(graph.input(shape)?),
        DType::BF16 => Ok(graph.input_bf16_as_f32(shape)?),
        DType::U8 => Ok(graph.input_with_dtype(shape, DType::U8)?),
        _ => UnsupportedParameterDTypeSnafu { dtype }.fail(),
    }
}

fn validate_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && !name.contains('.'),
        InvalidNameSnafu { name }
    );
    Ok(())
}

/// Interpret parameter effects as declarations and return the frozen schema.
pub fn init<T>(body: impl FnOnce(&mut Cx) -> Result<T>) -> Result<(ParamSchema, T)> {
    let mut cx = Cx::init();
    let result = body(&mut cx)?;
    cx.finish_rngs()?;
    Ok((cx.into_schema(), result))
}

/// Interpret parameter effects as reads from `schema` and retain traced outputs.
pub fn apply<T: ModelOutputs>(
    schema: &ParamSchema,
    body: impl FnOnce(&mut Cx) -> Result<T>,
) -> Result<AppliedModel> {
    apply_with_resident(schema, HashSet::new(), body)
}

/// Interpret selected parameters as resident state rather than ABI inputs.
pub fn apply_resident<T: ModelOutputs>(
    schema: &ParamSchema,
    selection: &ParameterSelection,
    body: impl FnOnce(&mut Cx) -> Result<T>,
) -> Result<AppliedModel> {
    ensure!(
        selection.schema().same_identity(schema),
        SelectionSchemaMismatchSnafu
    );
    apply_with_resident(schema, selection.ids().iter().copied().collect(), body)
}

fn apply_with_resident<T: ModelOutputs>(
    schema: &ParamSchema,
    resident: HashSet<ParameterId>,
    body: impl FnOnce(&mut Cx) -> Result<T>,
) -> Result<AppliedModel> {
    let mut cx = Cx::apply(schema.clone(), resident);
    let outputs = body(&mut cx)?.into_tensors();
    cx.finish_rngs()?;
    cx.finish_apply()?;
    let parameters = cx.parameter_tensors();
    let states = cx.state_slots(schema);
    let resident_parameters = cx.resident_parameter_slots(schema);
    Ok(AppliedModel::new(
        cx.graph,
        states,
        outputs,
        parameters,
        resident_parameters,
        schema.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{Buffer, CacheLimits, Client, ClientOptions, Compiler, Conv2dOptions};

    fn classifier(cx: &mut Cx) -> Result<Tensor> {
        let input = cx.input(&[2, 4])?;
        let mut scope = cx.scope("head")?;
        let weight = scope.param("weight", &[4, 3])?;
        Ok(input.matmul(&weight)?)
    }

    #[test]
    fn init_and_apply_share_one_model_body_and_schema() {
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
    fn scope_guards_restore_paths_and_validate_atomically() {
        let (schema, _) = init(|cx| {
            {
                let mut block = cx.scope_path(["encoder", "0"])?;
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
    fn byte_inputs_can_be_preprocessed_inside_the_model_program() {
        fn preprocess(cx: &mut Cx) -> Result<Tensor> {
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
        fn product(cx: &mut Cx) -> Result<Tensor> {
            let body = cx.scope("body")?.param("weight", &[2])?;
            let head = cx.scope("head")?.param("weight", &[2])?;
            Ok(body.mul(&head)?.sum(&[0], false)?)
        }

        let (schema, _) = init(product).unwrap();
        let applied = apply(&schema, product).unwrap();
        let head = schema.select_under("head");
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
    fn apply_rejects_changed_or_missing_parameter_effects() {
        let (schema, _) = init(|cx| cx.param("weight", &[2, 3])).unwrap();
        let changed = match apply(&schema, |cx| {
            cx.param("weight", &[3, 2])?;
            Ok(())
        }) {
            Ok(_) => panic!("changed parameter shape unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(changed, Error::IncompatibleParameter { .. }));

        let missing = match apply(&schema, |_cx| Ok(())) {
            Ok(_) => panic!("missing parameter effect unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(missing, Error::UnreadParameter { .. }));
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
    fn apply_rejects_changed_or_extra_input_effects() {
        let (schema, _) = init(classifier).unwrap();
        let changed = match apply(&schema, |cx| {
            let input = cx.input(&[3, 4])?;
            let weight = cx.scope("head")?.param("weight", &[4, 3])?;
            Ok(input.matmul(&weight)?)
        }) {
            Ok(_) => panic!("changed input ABI unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(changed, Error::IncompatibleInput { index: 0 }));

        let extra = match apply(&schema, |cx| {
            let input = cx.input(&[2, 4])?;
            let _unused = cx.input(&[1])?;
            let weight = cx.scope("head")?.param("weight", &[4, 3])?;
            Ok(input.matmul(&weight)?)
        }) {
            Ok(_) => panic!("extra input ABI unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(extra, Error::UnexpectedInput { index: 1 }));
    }

    #[test]
    fn apply_rejects_effect_reordering() {
        let (schema, _) = init(classifier).unwrap();
        let error = match apply(&schema, |cx| {
            let weight = cx.scope("head")?.param("weight", &[4, 3])?;
            let input = cx.input(&[2, 4])?;
            Ok(input.matmul(&weight)?)
        }) {
            Ok(_) => panic!("reordered effects unexpectedly applied"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::EffectMismatch { index: 0 }));
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
            cx.layer("head")?.linear(3).apply(&input)
        })
        .unwrap();
        assert_eq!(output.shape(), [2, 3]);
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.parameters()[0].shape(), [3, 4]);
        assert_eq!(schema.parameters()[1].path(), "head.bias");
        assert_eq!(schema.parameters()[1].shape(), [3]);
    }

    #[test]
    fn program_owns_the_model_body_and_layer_builders_own_scopes() {
        let model = Model::new(|cx: &mut Cx| {
            let input = cx.input(&[2, 4])?;
            let hidden = cx
                .layer("hidden")?
                .linear(8)
                .bias(false)
                .apply(&input)?
                .relu()?;
            cx.layer("head")?.linear(3).apply(&hidden)
        });

        let (schema, applied) = model.trace().unwrap();

        assert_eq!(schema.parameters()[0].path(), "hidden.weight");
        assert_eq!(schema.parameters()[0].shape(), [8, 4]);
        assert_eq!(schema.parameters()[1].path(), "head.weight");
        assert_eq!(schema.parameters()[1].shape(), [3, 8]);
        assert_eq!(schema.parameters()[2].path(), "head.bias");
        assert_eq!(applied.outputs()[0].shape(), [2, 3]);
    }

    #[test]
    fn one_context_composes_parameters_and_resident_state() {
        let model = Model::new(|cx: &mut Cx| {
            let input = cx.input(&[2, 4])?;
            let output = cx.layer("head")?.linear(3).apply(&input)?;
            let count = cx.state("steps", &[], DType::I32)?;
            let next = count.read(cx)?.wrapping_add_scalar(1)?;
            count.write(cx, &next)?;
            let noise = cx.rng("sampling")?.normal_f32(output.shape())?;
            Ok(output.add(&noise.mul_scalar(0.0)?)?)
        });

        let (schema, applied) = model.trace().unwrap();
        assert_eq!(schema.parameters()[0].path(), "head.weight");
        assert_eq!(schema.states()[0].path(), "steps");
        assert_eq!(schema.states()[1].path(), "sampling.key0");
        assert!(applied.is_stateful());
        assert!(applied.prepare().is_err());
        let prepared = applied.prepare_stateful().unwrap();
        let (_, steps) = applied.states().next().unwrap();
        assert_eq!(prepared.state_type(steps).unwrap(), (DType::I32, vec![]));
    }

    #[test]
    fn transforms_append_named_resident_state() {
        let (_, mut applied) = Model::new(|cx: &mut Cx| cx.input(&[2])).trace().unwrap();
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
        let definition = Model::new(|cx: &mut Cx, input: Tensor| {
            cx.layer("head")?.linear(2).bias(false).apply(&input)
        })
        .inputs(ModelInput::new([1, 3]));
        let (schema, selection, applied) = definition
            .trace_resident(|schema| schema.select_under("head"))
            .unwrap();

        assert!(applied.is_stateful());
        assert_eq!(selection.len(), 1);
        assert_eq!(schema.parameters().len(), 1);
        assert_eq!(applied.resident_parameters().count(), 1);
        let prepared = applied.prepare_stateful().unwrap();
        assert_eq!(prepared.input_indices(), [0]);
        let (_, spec, slot) = applied.resident_parameters().next().unwrap();
        assert_eq!(spec.path(), "head.weight");
        assert_eq!(prepared.state_type(slot).unwrap(), (DType::F32, vec![2, 3]));

        let quantized = Model::new(|cx: &mut Cx| {
            Ok(cx
                .param_dtype("weight", &[2], DType::U8)?
                .cast(DType::F32)?)
        });
        let schema = quantized.init().unwrap();
        let applied = quantized
            .apply_resident(&schema, &schema.select_all())
            .unwrap();
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
        let definition = Model::new(|cx: &mut Cx, input: Tensor| {
            cx.layer("head")?.linear(2).bias(false).apply(&input)
        })
        .inputs(ModelInput::new([1, 3]));
        let schema = definition.init().unwrap();
        let applied = definition
            .apply_resident(&schema, &schema.select_all())
            .unwrap();
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

        let snapshot = applied.take_session(session.into_raw()).unwrap();
        let nonresident = definition
            .apply_resident(&schema, &schema.select_all().matching(|_, _| false))
            .unwrap();
        let nonresident_program = nonresident.compile_stateful(&mut compiler).unwrap();
        assert!(matches!(
            snapshot.restore_model(nonresident_program.session().into_raw_builder()),
            Err(Error::UnexpectedResidentParameter { .. })
        ));
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn unified_context_state_and_rng_execute_as_one_program() {
        let model = Model::new(|cx: &mut Cx| {
            let steps = cx.state("steps", &[], DType::I32)?;
            let next = steps.read(cx)?.wrapping_add_scalar(1)?;
            steps.write(cx, &next)?;
            let draw = cx.rng("sampling")?.uniform_f32(&[4])?;
            Ok([draw, next])
        });
        let (_, applied) = model.trace().unwrap();
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
        let model = |cx: &mut Cx| {
            let input = cx.input(&[1, 8, 8, 4])?;
            cx.layer("conv_in")?
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
        let model = |cx: &mut Cx| {
            let input = cx.input(&[1, 8, 8, 32])?;
            cx.layer("norm")?.group_norm(8).apply(&input)
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
            cx.layer("norm")?.layer_norm(1).apply(&input)
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

        let model = |cx: &mut Cx| {
            let input = cx.input(&[2, 3])?;
            Ok(cx
                .layer("head")?
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
