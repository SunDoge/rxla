//! Applied model snapshots, compilation, and runtime argument binding.

use super::*;
use rxla_core::{
    Buffer, Compiler, DType, Executable, LoweredProgram, PreparedStateGraph, Session, StateGraph,
    StateProgram, StateSlot, Tracer,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One apply trace plus its immutable Pliron program builder.
#[derive(Clone)]
pub struct AppliedModel {
    graph: Tracer,
    state_graph: StateGraph,
    states: Vec<(String, StateSlot)>,
    outputs: Vec<Tensor>,
    parameters: Vec<Tensor>,
    resident_parameters: Vec<Option<StateSlot>>,
    schema: ParamSchema,
}

/// Validated runtime buffers in the ABI order required by an [`AppliedModel`].
///
/// Construct this through [`AppliedModel::bind`], then pass
/// [`Self::as_slice`] to [`Executable::execute`].
pub struct ModelArguments<'a> {
    values: Vec<&'a Buffer>,
}

/// A compiled stateless model that retains its named parameter ABI and output structure.
pub struct CompiledModel {
    model: AppliedModel,
    executable: Arc<Executable>,
}

/// A self-contained compiled model with owned, shared device-buffer handles.
pub struct BoundModel {
    model: AppliedModel,
    executable: Arc<Executable>,
    parameters: Vec<Option<Buffer>>,
}

/// Named initialization for a compiled unified stateful model.
pub struct ModelSessionBuilder<'a> {
    model: &'a AppliedModel,
    program: &'a StateProgram,
    overrides: BTreeMap<String, Buffer>,
    parameter_overrides: BTreeMap<String, Buffer>,
}

/// Owned device buffers extracted from one model session by canonical path.
///
/// This is an in-process, same-client handoff rather than a serialized
/// checkpoint. It can restore a compatible inference/training trace while
/// leaving source-only transform state (for example optimizer moments) unused.
pub struct ModelSessionBuffers {
    parameters: BTreeMap<String, Buffer>,
    states: BTreeMap<String, Buffer>,
}

/// One optimizer/transform-owned resident state slot appended after model tracing.
pub struct TransformState {
    path: String,
    slot: StateSlot,
    value: Tensor,
}

impl TransformState {
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Start-of-step symbolic value used to construct a transformation.
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// Resident slot initialized and inspected by a compiled model session.
    pub fn slot(&self) -> &StateSlot {
        &self.slot
    }
}

impl<'a> ModelArguments<'a> {
    pub fn as_slice(&self) -> &[&'a Buffer] {
        &self.values
    }
}

impl CompiledModel {
    /// Access the low-level executable for profiler and backend integrations.
    pub fn executable(&self) -> &Executable {
        &self.executable
    }

    /// Bind named parameters once for repeated typed model execution.
    pub fn bind_parameters<'parameters>(
        &self,
        parameters: impl IntoIterator<Item = (&'parameters str, &'parameters Buffer)>,
    ) -> Result<BoundModel> {
        let parameters = self
            .model
            .order_parameters(parameters)?
            .into_iter()
            .map(|value| value.cloned())
            .collect();
        Ok(BoundModel {
            model: self.model.clone(),
            executable: Arc::clone(&self.executable),
            parameters,
        })
    }

    /// Validate inputs and parameters, execute, and reconstruct the requested output structure.
    pub fn run<'a, I, O>(
        &self,
        inputs: I,
        parameters: impl IntoIterator<Item = (&'a str, &'a Buffer)>,
    ) -> Result<O>
    where
        I: ModelInputValues<'a>,
        O: ModelOutputValues,
    {
        let arguments = self.model.bind(inputs, parameters)?;
        let outputs = self.executable.execute(arguments.as_slice())?;
        self.model.decode_outputs(outputs)
    }
}

impl BoundModel {
    /// Execute with changing inputs while reusing validated parameter bindings.
    pub fn run<'inputs, I, O>(&self, inputs: I) -> Result<O>
    where
        I: ModelInputValues<'inputs>,
        O: ModelOutputValues,
    {
        let inputs = inputs.into_values();
        let parameters = self
            .parameters
            .iter()
            .map(Option::as_ref)
            .collect::<Vec<_>>();
        let arguments = self.model.bind_ordered(&inputs, &parameters)?;
        let outputs = self.executable.execute(arguments.as_slice())?;
        self.model.decode_outputs(outputs)
    }

    pub fn executable(&self) -> &Executable {
        &self.executable
    }
}

impl AppliedModel {
    pub(crate) fn new(
        state_graph: StateGraph,
        states: Vec<(String, StateSlot)>,
        outputs: Vec<Tensor>,
        parameters: Vec<Tensor>,
        resident_parameters: Vec<Option<StateSlot>>,
        schema: ParamSchema,
    ) -> Self {
        let graph = state_graph.tracer();
        Self {
            graph,
            state_graph,
            states,
            outputs,
            parameters,
            resident_parameters,
            schema,
        }
    }

    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }

    /// Decode one ordinary model execution into a typed buffer structure.
    pub fn decode_outputs<O: ModelOutputValues>(&self, outputs: Vec<Buffer>) -> Result<O> {
        ensure!(
            outputs.len() == self.outputs.len(),
            OutputCountSnafu {
                expected: self.outputs.len(),
                actual: outputs.len(),
            }
        );
        let mut outputs = outputs.into_iter();
        let value = O::take_from(&mut outputs).context(OutputStructureSnafu)?;
        ensure!(outputs.next().is_none(), OutputStructureSnafu);
        Ok(value)
    }

    pub fn states(&self) -> impl ExactSizeIterator<Item = (&str, &StateSlot)> {
        self.states.iter().map(|(path, slot)| (path.as_str(), slot))
    }

    /// Resident parameter identities in schema order.
    pub fn resident_parameters(
        &self,
    ) -> impl Iterator<Item = (ParameterId, &ParameterSpec, &StateSlot)> {
        self.schema
            .parameters()
            .iter()
            .enumerate()
            .filter_map(|(index, spec)| {
                self.resident_parameters[index]
                    .as_ref()
                    .map(|slot| (ParameterId::from_index(index), spec, slot))
            })
    }

    /// Borrow canonical resident parameter names and their current session buffers.
    ///
    /// This is suitable for checkpoint export. A session compiled from another
    /// trace is rejected through state-slot identity before any buffers are
    /// returned.
    pub fn resident_parameter_buffers<'model, 'session>(
        &'model self,
        session: &'session Session,
    ) -> Result<Vec<(&'model str, &'session Buffer)>> {
        self.resident_parameters()
            .map(|(_, spec, slot)| Ok((spec.path(), session.state(slot)?)))
            .collect()
    }

    /// Borrow named model and transform state from a compatible session.
    pub fn state_buffers<'model, 'session>(
        &'model self,
        session: &'session Session,
    ) -> Result<Vec<(&'model str, &'session Buffer)>> {
        self.states
            .iter()
            .map(|(path, slot)| Ok((path.as_str(), session.state(slot)?)))
            .collect()
    }

    /// Consume a compatible session into path-addressed model buffers.
    pub fn take_session(&self, session: Session) -> Result<ModelSessionBuffers> {
        // Validate every expected identity before consuming the session.
        self.resident_parameter_buffers(&session)?;
        self.state_buffers(&session)?;
        let mut by_slot = session
            .into_state()
            .into_iter()
            .map(|(slot, buffer)| (slot.identity(), buffer))
            .collect::<BTreeMap<_, _>>();
        let parameters =
            self.resident_parameters()
                .map(|(_, spec, slot)| {
                    let buffer = by_slot.remove(&slot.identity()).with_context(|| {
                        InvalidDefinitionSnafu {
                            message: format!(
                                "session is missing resident parameter {:?}",
                                spec.path()
                            ),
                        }
                    })?;
                    Ok((spec.path().to_owned(), buffer))
                })
                .collect::<Result<_>>()?;
        let states =
            self.states
                .iter()
                .map(|(path, slot)| {
                    let buffer = by_slot.remove(&slot.identity()).with_context(|| {
                        InvalidDefinitionSnafu {
                            message: format!("session is missing state {path:?}"),
                        }
                    })?;
                    Ok((path.clone(), buffer))
                })
                .collect::<Result<_>>()?;
        ensure!(
            by_slot.is_empty(),
            InvalidDefinitionSnafu {
                message: "session contains state absent from its applied model"
            }
        );
        Ok(ModelSessionBuffers { parameters, states })
    }

    /// Validate that a selection belongs to this trace and every member uses
    /// resident storage. Performs no graph mutation.
    pub fn validate_resident_parameters(&self, selection: &ParameterSelection) -> Result<()> {
        ensure!(
            selection.schema().same_identity(&self.schema),
            SelectionSchemaMismatchSnafu
        );
        for (id, spec) in selection.parameters() {
            ensure!(
                self.resident_parameters[id.index()].is_some(),
                ParameterNotResidentSnafu { path: spec.path() }
            );
        }
        Ok(())
    }

    pub fn is_stateful(&self) -> bool {
        !self.states.is_empty() || self.resident_parameters.iter().any(Option::is_some)
    }

    /// The frozen effect schema used to produce this program.
    pub fn schema(&self) -> &ParamSchema {
        &self.schema
    }

    /// Clone the cheap graph handles for parameters selected for a transform.
    pub fn parameter_tensors(&self, selection: &ParameterSelection) -> Result<Vec<Tensor>> {
        ensure!(
            selection.schema().same_identity(&self.schema),
            SelectionSchemaMismatchSnafu
        );
        Ok(selection
            .ids()
            .iter()
            .map(|id| self.parameters[id.index()].clone())
            .collect())
    }

    /// Append named resident state owned by a graph transformation.
    ///
    /// Unlike a model `Cx::state` effect, this does not modify `ParamSchema` or
    /// require changing the model body. It becomes part of this applied model's
    /// stateful program and session layout immediately.
    pub fn transform_state(
        &mut self,
        path: impl Into<String>,
        shape: &[i64],
        dtype: DType,
    ) -> Result<TransformState> {
        let path = path.into();
        ensure!(
            !self.states.iter().any(|(existing, _)| existing == &path),
            DuplicateTransformStateSnafu { path: path.clone() }
        );
        let slot = self.state_graph.state_named(&path, shape, dtype)?;
        let value = self.state_graph.read(&slot)?;
        self.states.push((path.clone(), slot.clone()));
        Ok(TransformState { path, slot, value })
    }

    /// Atomically record next values for transform-owned state.
    pub fn write_transform_states(&mut self, updates: &[(&TransformState, &Tensor)]) -> Result<()> {
        Ok(self.state_graph.write_many(
            &updates
                .iter()
                .map(|(state, value)| (&state.slot, *value))
                .collect::<Vec<_>>(),
        )?)
    }

    /// Atomically record next values for selected resident parameters.
    ///
    /// The selection and values use schema order. Every selected parameter must
    /// have been made resident by `Model::apply_resident`; validation completes
    /// before any symbolic slot is changed.
    pub fn write_resident_parameters(
        &mut self,
        selection: &ParameterSelection,
        values: &[Tensor],
    ) -> Result<()> {
        self.validate_resident_parameters(selection)?;
        ensure!(
            selection.len() == values.len(),
            ParameterUpdateCountSnafu {
                expected: selection.len(),
                actual: values.len(),
            }
        );
        let updates = selection
            .parameters()
            .zip(values)
            .map(|((id, _), value)| {
                let slot = self.resident_parameters[id.index()]
                    .as_ref()
                    .expect("resident selection validated");
                Ok((slot, value))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.state_graph.write_many(&updates)?)
    }

    /// Lower model outputs while preserving the frozen schema ABI exactly.
    pub fn prepare(&self) -> Result<LoweredProgram> {
        ensure!(
            !self.is_stateful(),
            InvalidDefinitionSnafu {
                message: "stateful models must use prepare_stateful"
            }
        );
        Ok(self.graph.prepare_many(&self.outputs)?)
    }

    pub fn prepare_stateful(&self) -> Result<PreparedStateGraph> {
        Ok(self.state_graph.prepare(&self.outputs)?)
    }

    /// Compile a stateless model while retaining its typed model ABI.
    pub fn compile(&self, compiler: &mut Compiler) -> Result<CompiledModel> {
        Ok(CompiledModel {
            model: self.clone(),
            executable: self.compile_executable(compiler)?,
        })
    }

    /// Compile to a low-level executable without retaining model binding metadata.
    pub fn compile_executable(&self, compiler: &mut Compiler) -> Result<Arc<Executable>> {
        ensure!(
            !self.is_stateful(),
            InvalidDefinitionSnafu {
                message: "stateful models must use compile_stateful"
            }
        );
        Ok(compiler.compile_many(&self.graph, &self.outputs)?)
    }

    pub fn compile_stateful(&self, compiler: &mut Compiler) -> Result<StateProgram> {
        Ok(self.state_graph.compile(compiler, &self.outputs)?)
    }

    /// Lower graph-local transform outputs while retaining every final resident
    /// state version as a hidden root.
    pub fn prepare_stateful_tensors(&self, outputs: &[Tensor]) -> Result<PreparedStateGraph> {
        Ok(self.state_graph.prepare(outputs)?)
    }

    /// Compile graph-local transform outputs and all resident state transitions
    /// as one program. This is the state-aware counterpart to
    /// [`Self::compile_tensors`].
    pub fn compile_stateful_tensors(
        &self,
        compiler: &mut Compiler,
        outputs: &[Tensor],
    ) -> Result<StateProgram> {
        Ok(self.state_graph.compile(compiler, outputs)?)
    }

    pub fn session<'a>(&'a self, program: &'a StateProgram) -> ModelSessionBuilder<'a> {
        ModelSessionBuilder {
            model: self,
            program,
            overrides: BTreeMap::new(),
            parameter_overrides: BTreeMap::new(),
        }
    }

    /// Lower graph-local outputs produced by a model transformation.
    pub fn prepare_tensors(&self, outputs: &[Tensor]) -> Result<LoweredProgram> {
        ensure!(
            !self.is_stateful(),
            InvalidDefinitionSnafu {
                message: "stateful transforms require a state-aware training path"
            }
        );
        Ok(self.graph.prepare_many(outputs)?)
    }

    /// Compile graph-local outputs produced by a model transformation.
    pub fn compile_tensors(
        &self,
        compiler: &mut Compiler,
        outputs: &[Tensor],
    ) -> Result<Arc<Executable>> {
        ensure!(
            !self.is_stateful(),
            InvalidDefinitionSnafu {
                message: "stateful transforms require a state-aware training path"
            }
        );
        Ok(compiler.compile_many(&self.graph, outputs)?)
    }

    /// Assemble positional inputs and path-addressed parameters into the model ABI.
    pub fn bind<'a, I>(
        &self,
        inputs: I,
        parameters: impl IntoIterator<Item = (&'a str, &'a Buffer)>,
    ) -> Result<ModelArguments<'a>>
    where
        I: ModelInputValues<'a>,
    {
        let inputs = inputs.into_values();
        self.validate_inputs(&inputs)?;
        let parameters = self.order_parameters(parameters)?;
        self.assemble(&inputs, &parameters)
    }

    fn order_parameters<'a>(
        &self,
        parameters: impl IntoIterator<Item = (&'a str, &'a Buffer)>,
    ) -> Result<Vec<Option<&'a Buffer>>> {
        let mut named = BTreeMap::new();
        for (path, buffer) in parameters {
            let spec = self
                .schema
                .get(path)
                .with_context(|| UnknownParameterSnafu { path })?;
            ensure!(
                named.insert(path, buffer).is_none(),
                DuplicateBindingSnafu { path }
            );
            validate_buffer(buffer, spec.shape(), spec.dtype(), "parameter", path)?;
        }
        self.schema
            .parameters()
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                if self.resident_parameters[index].is_some() {
                    ensure!(
                        !named.contains_key(spec.path()),
                        InvalidDefinitionSnafu {
                            message: format!(
                                "resident parameter {:?} must be initialized through a session",
                                spec.path()
                            )
                        }
                    );
                    return Ok(None);
                }
                named
                    .get(spec.path())
                    .copied()
                    .with_context(|| MissingBindingSnafu { path: spec.path() })
                    .map(Some)
            })
            .collect()
    }

    fn bind_ordered<'a>(
        &self,
        inputs: &[&'a Buffer],
        parameters: &[Option<&'a Buffer>],
    ) -> Result<ModelArguments<'a>> {
        self.validate_inputs(inputs)?;
        self.assemble(inputs, parameters)
    }

    fn validate_inputs(&self, inputs: &[&Buffer]) -> Result<()> {
        ensure!(
            inputs.len() == self.schema.inputs().len(),
            InputCountSnafu {
                expected: self.schema.inputs().len(),
                actual: inputs.len(),
            }
        );
        for (index, (buffer, spec)) in inputs.iter().zip(self.schema.inputs()).enumerate() {
            validate_buffer(buffer, spec.shape(), spec.dtype(), "input", index)?;
        }
        Ok(())
    }

    fn assemble<'a>(
        &self,
        inputs: &[&'a Buffer],
        parameters: &[Option<&'a Buffer>],
    ) -> Result<ModelArguments<'a>> {
        ensure!(
            parameters.len() == self.schema.parameters().len(),
            ParameterCountSnafu {
                expected: self.schema.parameters().len(),
                actual: parameters.len(),
            }
        );
        let mut values = Vec::with_capacity(self.schema.arguments().len());
        for argument in self.schema.arguments() {
            match *argument {
                ModelArgument::Input(index) => values.push(inputs[index]),
                ModelArgument::Parameter(index) => {
                    if let Some(parameter) = parameters[index] {
                        values.push(parameter);
                    }
                }
                ModelArgument::State(_) => {}
            }
        }
        Ok(ModelArguments { values })
    }

    pub fn into_parts(self) -> (Tracer, Vec<Tensor>) {
        (self.graph, self.outputs)
    }
}

impl ModelSessionBuffers {
    pub fn parameters(&self) -> impl ExactSizeIterator<Item = (&str, &Buffer)> {
        self.parameters
            .iter()
            .map(|(path, buffer)| (path.as_str(), buffer))
    }

    pub fn states(&self) -> impl ExactSizeIterator<Item = (&str, &Buffer)> {
        self.states
            .iter()
            .map(|(path, buffer)| (path.as_str(), buffer))
    }

    /// Move all buffers required by `builder` into its target model.
    ///
    /// The source and target resident parameter sets must match exactly, and
    /// every target named state must exist with a valid client/shape/dtype.
    /// Extra source state is deliberately ignored so a training snapshot can
    /// initialize an inference trace without carrying optimizer-only slots.
    pub fn restore_model<'model>(
        mut self,
        mut builder: ModelSessionBuilder<'model>,
    ) -> Result<ModelSessionBuilder<'model>> {
        let parameter_paths = builder
            .model
            .resident_parameters()
            .map(|(_, spec, _)| spec.path().to_owned())
            .collect::<Vec<_>>();
        for path in parameter_paths {
            let value = self
                .parameters
                .remove(&path)
                .with_context(|| MissingResidentParameterInitializerSnafu { path: &path })?;
            builder = builder.parameter(path, value)?;
        }
        if let Some(path) = self.parameters.keys().next() {
            return UnexpectedResidentParameterSnafu { path: path.clone() }.fail();
        }
        let state_paths = builder
            .model
            .states
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        for path in state_paths {
            let value = self
                .states
                .remove(&path)
                .with_context(|| InvalidDefinitionSnafu {
                    message: format!("session snapshot is missing target state {path:?}"),
                })?;
            builder = builder.state(path, value)?;
        }
        Ok(builder)
    }
}

impl ModelSessionBuilder<'_> {
    pub fn state(mut self, name: impl Into<String>, value: Buffer) -> Result<Self> {
        let name = name.into();
        let slot = self
            .model
            .states
            .iter()
            .find(|(path, _)| path == &name)
            .map(|(_, slot)| slot)
            .with_context(|| InvalidDefinitionSnafu {
                message: format!("unknown state {name:?}"),
            })?;
        validate_session_buffer(self.program, &value, slot, "state", &name)?;
        ensure!(
            self.overrides.insert(name.clone(), value).is_none(),
            InvalidDefinitionSnafu {
                message: format!("duplicate state initializer {name:?}")
            }
        );
        Ok(self)
    }

    /// Initialize several named model/transform states.
    pub fn states<K>(mut self, values: impl IntoIterator<Item = (K, Buffer)>) -> Result<Self>
    where
        K: Into<String>,
    {
        for (name, value) in values {
            self = self.state(name, value)?;
        }
        Ok(self)
    }

    /// Initialize one resident parameter by its canonical schema path.
    pub fn parameter(mut self, path: impl Into<String>, value: Buffer) -> Result<Self> {
        let path = path.into();
        let (_, _, slot) = self
            .model
            .resident_parameters()
            .find(|(_, spec, _)| spec.path() == path)
            .with_context(|| InvalidDefinitionSnafu {
                message: format!("unknown resident parameter {path:?}"),
            })?;
        validate_session_buffer(self.program, &value, slot, "resident parameter", &path)?;
        ensure!(
            self.parameter_overrides
                .insert(path.clone(), value)
                .is_none(),
            InvalidDefinitionSnafu {
                message: format!("duplicate resident parameter initializer {path:?}")
            }
        );
        Ok(self)
    }

    /// Initialize resident parameters from a checkpoint-style name/buffer map.
    pub fn parameters<K>(mut self, values: impl IntoIterator<Item = (K, Buffer)>) -> Result<Self>
    where
        K: Into<String>,
    {
        for (path, value) in values {
            self = self.parameter(path, value)?;
        }
        Ok(self)
    }

    pub fn rng_seed(self, name: &str, seed: u64) -> Result<Self> {
        self.rng_state(name, [seed as u32, (seed >> 32) as u32], 0)
    }

    pub fn rng_state(mut self, name: &str, key: [u32; 2], counter: u64) -> Result<Self> {
        for (suffix, word) in [
            ("key0", key[0]),
            ("key1", key[1]),
            ("counter_low", counter as u32),
            ("counter_high", (counter >> 32) as u32),
        ] {
            let path = format!("{name}.{suffix}");
            ensure!(
                self.model.states.iter().any(|(state, _)| state == &path),
                InvalidDefinitionSnafu {
                    message: format!("unknown RNG stream {name:?}")
                }
            );
            let value = self.program.client().buffer(&[], &[word as i32])?;
            ensure!(
                self.overrides.insert(path.clone(), value).is_none(),
                InvalidDefinitionSnafu {
                    message: format!("duplicate state initializer {path:?}")
                }
            );
        }
        Ok(self)
    }

    pub fn build(mut self) -> Result<Session> {
        let resident = self
            .model
            .resident_parameters()
            .map(|(_, spec, slot)| (spec.path().to_owned(), slot.clone()))
            .collect::<Vec<_>>();
        let mut initial = Vec::with_capacity(resident.len() + self.model.states.len());
        for (name, slot) in &resident {
            let value = self
                .parameter_overrides
                .remove(name)
                .with_context(|| MissingResidentParameterInitializerSnafu { path: name })?;
            initial.push((slot.clone(), value));
        }
        let mut ordinary = self
            .model
            .states
            .iter()
            .map(|(_, slot)| Ok((slot.clone(), self.program.zero_state_slot(slot)?)))
            .collect::<Result<Vec<_>>>()?;
        for ((name, _), (_, value)) in self.model.states.iter().zip(&mut ordinary) {
            if let Some(override_value) = self.overrides.remove(name) {
                *value = override_value;
            }
        }
        initial.extend(ordinary);
        ensure!(
            self.parameter_overrides.is_empty(),
            InvalidDefinitionSnafu {
                message: "unused resident parameter initializers"
            }
        );
        Ok(self.program.session(initial)?)
    }
}

fn validate_buffer(
    buffer: &Buffer,
    shape: &[i64],
    dtype: DType,
    kind: &'static str,
    identity: impl std::fmt::Display,
) -> Result<()> {
    ensure!(
        buffer.dimensions()? == shape,
        BufferShapeSnafu {
            kind,
            identity: identity.to_string(),
        }
    );
    ensure!(
        buffer.dtype()? == dtype,
        BufferDTypeSnafu {
            kind,
            identity: identity.to_string(),
        }
    );
    Ok(())
}

fn validate_session_buffer(
    program: &StateProgram,
    buffer: &Buffer,
    slot: &StateSlot,
    kind: &'static str,
    identity: impl std::fmt::Display,
) -> Result<()> {
    let identity = identity.to_string();
    ensure!(
        buffer.belongs_to(program.client()),
        BufferClientSnafu {
            kind,
            identity: identity.clone(),
        }
    );
    let (dtype, shape) = program.state_type(slot)?;
    validate_buffer(buffer, &shape, dtype, kind, identity)
}
