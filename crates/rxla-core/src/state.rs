//! Stateful facade, functional compilation, and non-donating session execution.
use super::*;
use std::{rc::Rc, sync::Arc};

/// Identity of an F32 or I32 state slot. Clones identify the same slot, not copies.
#[derive(Clone)]
pub struct StateSlot {
    owner: Arc<()>,
    index: usize,
}
impl StateSlot {
    #[doc(hidden)]
    pub fn identity(&self) -> (usize, usize) {
        (Arc::as_ptr(&self.owner) as usize, self.index)
    }
}

/// Identity and symbolic F32 value of a parameter in one state graph.
/// Frozen inputs may use BF16 storage with explicit conversion to F32.
/// Cloning preserves identity (e.g. tied weights); it does not register another
/// input or copy device data. This does not imply automatic differentiation.
#[derive(Clone)]
pub struct Parameter {
    owner: Arc<()>,
    storage: ParameterStorage,
    storage_dtype: DType,
    tensor: Tensor,
}

/// Process-local identity of a parameter allocation.
///
/// IDs compare tied/shared parameters without exposing graph pointers or input
/// indices. They are not stable across processes and must not be serialized.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParameterId((usize, bool, usize));
#[derive(Clone)]
enum ParameterStorage {
    Input(usize),
    State(StateSlot),
}
impl Parameter {
    /// Runtime buffer dtype, which may differ from the symbolic F32 value.
    /// Frozen BF16 inputs retain BF16 storage; trainable parameters are F32.
    pub fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }
    // Owners stay alive through Parameter clones held by the collector. This key
    // is process-local identity only, never a serialized ID or compilation key.
    pub fn id(&self) -> ParameterId {
        let (state, index) = match &self.storage {
            ParameterStorage::Input(index) => (false, *index),
            ParameterStorage::State(slot) => (true, slot.index),
        };
        ParameterId((Arc::as_ptr(&self.owner) as usize, state, index))
    }
    /// Initial symbolic value for this graph execution. A state-backed parameter
    /// holds the start-of-step version, not a live read of subsequent writes
    /// recorded into StateGraph. Each session call supplies its latest buffer.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    /// Resident state identity for a trainable parameter, or None for a fixed/
    /// dynamic input parameter. Use this for initialization and optimizer writes.
    pub fn state_slot(&self) -> Option<&StateSlot> {
        match &self.storage {
            ParameterStorage::Input(_) => None,
            ParameterStorage::State(slot) => Some(slot),
        }
    }
}
#[derive(Clone, Copy)]
enum Argument {
    Input(usize),
    State(usize),
}

#[derive(Clone)]
enum InputBinding {
    Unused,
    Dynamic(usize),
    Fixed(Rc<Buffer>),
}

/// Records symbolic state versions without modifying live runtime buffers.
pub struct StateGraph {
    graph: Graph,
    owner: Arc<()>,
    slots: Vec<Tensor>,
    arguments: Vec<Argument>,
    input_count: usize,
}

impl Default for StateGraph {
    fn default() -> Self {
        Self {
            graph: Graph::default(),
            owner: Arc::new(()),
            slots: Vec::new(),
            arguments: Vec::new(),
            input_count: 0,
        }
    }
}
impl StateGraph {
    /// Borrow the ordinary dataflow view of this trace. State-aware callers
    /// must still compile through `StateGraph` so final writes remain roots.
    pub fn tracer(&self) -> Tracer {
        Tracer {
            graph: self.graph.clone(),
        }
    }
    /// Register a runtime F32 input with identity-based fixed binding support.
    /// The value remains a runtime argument, never a graph constant/cache key.
    pub fn parameter(&mut self, dims: &[i64]) -> Result<Parameter> {
        let index = self.input_count;
        let tensor = self.input(dims)?;
        Ok(Parameter {
            owner: self.owner.clone(),
            storage: ParameterStorage::Input(index),
            storage_dtype: DType::F32,
            tensor,
        })
    }
    /// Register frozen BF16 input storage with an F32 symbolic value usable by
    /// existing modules. Bind exact BF16 buffers through `bind_parameters`.
    /// `parameter_type` reports storage dtype, while `tensor()` is the explicit
    /// F32 conversion. No trainable state or implicit BF16 derivative is added;
    /// see `Tracer::input_bf16_as_f32` for the differentiation boundary.
    pub fn parameter_bf16_as_f32(&mut self, dims: &[i64]) -> Result<Parameter> {
        let index = self.input_count;
        let tensor = self.input_bf16_as_f32(dims)?;
        Ok(Parameter {
            owner: self.owner.clone(),
            storage: ParameterStorage::Input(index),
            storage_dtype: DType::BF16,
            tensor,
        })
    }
    /// Register an F32 resident parameter usable by existing modules and grad.
    /// Initialize its `state_slot()` through program.session, and explicitly
    /// record optimizer writes to that slot. No initialization or optimizer is
    /// implicit. The handle captures the start-of-step tensor; later graph writes
    /// do not retarget already constructed module expressions.
    pub fn trainable_parameter(&mut self, dims: &[i64]) -> Result<Parameter> {
        let slot = self.state(dims)?;
        let tensor = self.read(&slot)?;
        Ok(Parameter {
            owner: self.owner.clone(),
            storage: ParameterStorage::State(slot),
            storage_dtype: DType::F32,
            tensor,
        })
    }
    pub fn input(&mut self, dims: &[i64]) -> Result<Tensor> {
        let tensor = self.graph.input(dims)?;
        self.arguments.push(Argument::Input(self.input_count));
        self.input_count += 1;
        Ok(tensor)
    }
    /// BF16 visible input with an explicit F32 graph conversion. Bind an exact
    /// BF16 buffer via Session::bind_inputs to reuse frozen weights. This does
    /// not register a trainable parameter or BF16 state slot; see Graph's method.
    pub fn input_bf16_as_f32(&mut self, dims: &[i64]) -> Result<Tensor> {
        let tensor = self.graph.input_bf16_as_f32(dims)?;
        self.arguments.push(Argument::Input(self.input_count));
        self.input_count += 1;
        Ok(tensor)
    }
    pub fn input_i32_scalar(&mut self) -> Result<Tensor> {
        self.input_i32(&[])
    }
    pub fn input_i32(&mut self, dims: &[i64]) -> Result<Tensor> {
        let index = self.graph.input_i32(dims)?;
        self.arguments.push(Argument::Input(self.input_count));
        self.input_count += 1;
        Ok(index)
    }
    #[doc(hidden)]
    pub fn input_with_dtype(&mut self, dims: &[i64], dtype: DType) -> Result<Tensor> {
        let tensor = self.graph.input_dtype(dims, dtype)?;
        self.arguments.push(Argument::Input(self.input_count));
        self.input_count += 1;
        Ok(tensor)
    }
    pub(crate) fn input_type(&mut self, ty: &TensorType) -> Result<Tensor> {
        let tensor = self.graph.input_dtype(&ty.dims, ty.dtype)?;
        self.arguments.push(Argument::Input(self.input_count));
        self.input_count += 1;
        Ok(tensor)
    }
    pub(crate) fn append_dataflow(
        &self,
        op: Op,
        operands: &[Tensor],
        ty: &TensorType,
    ) -> Result<Tensor> {
        if operands
            .iter()
            .any(|operand| !Arc::ptr_eq(&self.graph.0, &operand.graph().0))
        {
            return Err(err("imported operation belongs to another graph"));
        }
        let value =
            self.graph
                .node(op, operands.iter().map(Tensor::node_id).collect(), &ty.dims)?;
        if value.dtype() != ty.dtype {
            return Err(err("imported operation changed result dtype"));
        }
        Ok(value)
    }
    pub fn constant(&self, dims: &[i64], values: &[f32]) -> Result<Tensor> {
        self.graph.constant(dims, values)
    }
    pub fn scalar_i32(&self, value: i32) -> Result<Tensor> {
        self.graph.scalar_i32(value)
    }
    pub fn constant_i32(&self, dims: &[i64], values: &[i32]) -> Result<Tensor> {
        self.graph.constant_i32(dims, values)
    }
    /// Graph-local coordinate generation; no user input or state slot is added.
    pub fn iota_i32(&self, dims: &[i64], axis: usize) -> Result<Tensor> {
        self.graph.iota_i32(dims, axis)
    }
    /// Static additive causal mask; registers neither a visible input nor state.
    pub fn causal_attention_mask(
        &self,
        queries: i64,
        keys: i64,
        query_offset: i64,
    ) -> Result<Tensor> {
        self.graph
            .causal_attention_mask(queries, keys, query_offset)
    }
    pub fn state(&mut self, dims: &[i64]) -> Result<StateSlot> {
        self.state_named(&format!("state.{}", self.slots.len()), dims, DType::F32)
    }
    /// Register resident I32 state (use [] for a scalar position counter).
    pub fn state_i32(&mut self, dims: &[i64]) -> Result<StateSlot> {
        self.state_named(&format!("state.{}", self.slots.len()), dims, DType::I32)
    }
    #[doc(hidden)]
    pub fn state_named(&mut self, path: &str, dims: &[i64], dtype: DType) -> Result<StateSlot> {
        let index = self.slots.len();
        let value = self.graph.state_input(dims, dtype, index, path)?;
        Ok(self.register_state(value))
    }
    fn register_state(&mut self, value: Tensor) -> StateSlot {
        let index = self.slots.len();
        let value = self
            .graph
            .state_read(&value, index)
            .expect("new state input and read always share one graph and type");
        self.slots.push(value);
        self.arguments.push(Argument::State(index));
        StateSlot {
            owner: self.owner.clone(),
            index,
        }
    }
    pub fn read(&self, slot: &StateSlot) -> Result<Tensor> {
        validate_slot(&self.owner, self.slots.len(), slot)?;
        Ok(self.slots[slot.index].clone())
    }
    pub fn write(&mut self, slot: &StateSlot, value: &Tensor) -> Result<()> {
        self.record_updates(&[(slot, value.clone())])
    }
    /// Transform the current symbolic state and record its next version.
    /// The closure runs once now, during graph construction, not on each session
    /// execution. Only its tensor operations become device computation; ordinary
    /// Rust side effects are not captured. Invalid slots are rejected before the
    /// closure runs. Closure/validation errors leave the slot unchanged, but do
    /// not roll back appended graph nodes or external side effects.
    ///
    /// Returns the recorded value for use by later operations. Sequential updates
    /// observe earlier writes; use `write_many` for simultaneous updates instead.
    pub fn update(
        &mut self,
        slot: &StateSlot,
        transform: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let next = transform(&self.read(slot)?)?;
        self.write(slot, &next)?;
        Ok(next)
    }
    /// Transform F32 state and select its next version with a scalar F32 mask.
    /// Returns the selected version, not the unconditionally proposed value.
    /// The mask and slot are validated before invoking the closure. The closure
    /// runs once during construction even for a constant false mask; this is
    /// dataflow selection, not lazy control flow. Zero keeps the current state;
    /// nonzero (including NaN) selects the proposal. Errors preserve the slot,
    /// but appended nodes and ordinary Rust side effects are not rolled back.
    pub fn update_if(
        &mut self,
        condition: &Tensor,
        slot: &StateSlot,
        transform: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        self.validate_condition(condition)?;
        let next = transform(&self.read(slot)?)?;
        self.write_many_if(condition, &[(slot, next)])?;
        self.read(slot)
    }
    /// Atomically update a subset of symbolic state slots, e.g. a K/V pair.
    /// Every slot must be unique and every value must belong to this graph with
    /// the slot's existing shape. Errors leave all slot versions unchanged.
    /// Empty updates are a no-op; omitted slots retain their current versions.
    ///
    /// Values are built before this call, so swapping two previously read values
    /// is simultaneous. This changes the recorded graph, not live session state;
    /// tensor operations already built by the caller are not rolled back.
    pub fn write_many(&mut self, updates: &[(&StateSlot, &Tensor)]) -> Result<()> {
        self.record_updates(
            &updates
                .iter()
                .map(|(s, t)| (*s, (*t).clone()))
                .collect::<Vec<_>>(),
        )
    }
    pub(super) fn record_updates(&mut self, updates: &[(&StateSlot, Tensor)]) -> Result<()> {
        self.validate_updates(updates)?;
        for (slot, value) in updates {
            self.slots[slot.index] = self.graph.state_write(value, slot.index)?;
        }
        Ok(())
    }
    /// Record conditional mixed state updates using one graph-local scalar F32
    /// mask. Zero (including -0) keeps every old version; nonzero (including NaN)
    /// selects every proposed version. Unmentioned slots remain unchanged.
    ///
    /// Shapes, types, identities and duplicates are checked before changing any
    /// slot. Selection uses versions current at this call, so paired updates or
    /// swaps are simultaneous. This is dataflow selection, not lazy execution:
    /// proposed values still compute, and visible outputs are not gated. It does
    /// not catch execution errors or provide arbitrary side-effect rollback.
    pub fn write_many_if(
        &mut self,
        condition: &Tensor,
        updates: &[(&StateSlot, Tensor)],
    ) -> Result<()> {
        self.validate_condition(condition)?;
        self.validate_updates(updates)?;
        let selected = updates
            .iter()
            .map(|(slot, value)| {
                let mask = condition.broadcast_to(value.shape())?;
                let value = mask.select(value, &self.read(slot)?)?;
                Ok((*slot, value))
            })
            .collect::<Result<Vec<_>>>()?;
        self.record_updates(&selected)
    }
    fn validate_condition(&self, condition: &Tensor) -> Result<()> {
        if condition.dtype() != DType::F32 {
            return Err(err("state condition must be F32"));
        }
        if !Arc::ptr_eq(&self.graph.0, &condition.graph().0) {
            return Err(err("cross-graph state update condition"));
        }
        if !condition.shape().is_empty() {
            return Err(err("state update condition must be scalar"));
        }
        Ok(())
    }
    pub(super) fn validate_updates(&self, updates: &[(&StateSlot, Tensor)]) -> Result<()> {
        let mut seen = std::collections::HashSet::with_capacity(updates.len());
        for (slot, value) in updates {
            self.validate_write(slot, value)?;
            if !seen.insert(slot.index) {
                return Err(err("duplicate state update slot"));
            }
        }
        Ok(())
    }
    fn validate_write(&self, slot: &StateSlot, value: &Tensor) -> Result<()> {
        validate_slot(&self.owner, self.slots.len(), slot)?;
        if !Arc::ptr_eq(&self.graph.0, &value.graph().0) {
            return Err(err("cross-graph state value"));
        }
        if self.slots[slot.index].shape() != value.shape()
            || self.slots[slot.index].dtype() != value.dtype()
        {
            return Err(err("state shape or dtype cannot change"));
        }
        Ok(())
    }
    fn outputs_with_state(&self, outputs: &[Tensor]) -> Vec<Tensor> {
        let mut all = outputs.to_vec();
        all.extend(self.slots.iter().cloned());
        all
    }

    /// Snapshot visible outputs and all final state versions before native
    /// compilation. Later graph writes/registrations do not affect the snapshot.
    /// All declared inputs and original slot identities are retained.
    pub fn prepare(&self, outputs: &[Tensor]) -> Result<PreparedStateGraph> {
        self.prepare_state(outputs, false)
    }

    /// Prepare with unused input parameters removed, keeping hidden state
    /// updates as roots and retaining the complete state schema. Compiled plans
    /// expose compact visible input order through `StateProgram::input_indices`.
    pub fn prepare_pruned(&self, outputs: &[Tensor]) -> Result<PreparedStateGraph> {
        self.prepare_state(outputs, true)
    }

    fn prepare_state(&self, outputs: &[Tensor], pruned: bool) -> Result<PreparedStateGraph> {
        let all = self.outputs_with_state(outputs);
        let (graph, arguments) = if pruned {
            let (graph, parameters) = self.graph.prepare_pruned(&all)?;
            (
                graph,
                parameters.iter().map(|&i| self.arguments[i]).collect(),
            )
        } else {
            (self.graph.prepare_many(&all)?, self.arguments.clone())
        };
        let mut input_parameters = vec![None; self.input_count];
        for (parameter, argument) in arguments.iter().enumerate() {
            if let Argument::Input(index) = argument {
                input_parameters[*index] = Some(parameter);
            }
        }
        Ok(PreparedStateGraph {
            lowered: graph,
            owner: self.owner.clone(),
            arguments,
            input_parameters,
            input_count: self.input_count,
            visible_count: outputs.len(),
            types: all.iter().map(Tensor::ty).collect(),
        })
    }

    /// Final state versions become hidden output roots, including updates that
    /// do not contribute to user-visible results. Empty visible outputs are valid.
    pub fn compile(&self, compiler: &mut Compiler, outputs: &[Tensor]) -> Result<StateProgram> {
        let all_outputs = self.outputs_with_state(outputs);
        let executable = compiler.compile_graph_outputs(&self.graph, &all_outputs)?;
        Ok(StateProgram(Rc::new(Plan {
            executable,
            owner: self.owner.clone(),
            arguments: self.arguments.clone(),
            input_parameters: input_parameters(&self.arguments)
                .into_iter()
                .map(Some)
                .collect(),
            input_count: self.input_count,
            visible_count: outputs.len(),
            types: all_outputs.iter().map(Tensor::ty).collect(),
        })))
    }

    /// Opt-in input-ABI pruning. Only inputs reachable from visible outputs or
    /// final state versions remain required. All state slots and identities are
    /// retained, so state transfer/checkpoints remain schema-compatible.
    /// Original visible input numbers remain binding identities; inspect
    /// `StateProgram::input_indices` for the compact runtime order. Unlike
    /// `compile`, unused fixed parameters cannot be bound to this program.
    pub fn compile_pruned(
        &self,
        compiler: &mut Compiler,
        outputs: &[Tensor],
    ) -> Result<StateProgram> {
        let all_outputs = self.outputs_with_state(outputs);
        let (executable, parameters) =
            compiler.compile_graph_outputs_pruned(&self.graph, &all_outputs)?;
        let arguments: Vec<_> = parameters.iter().map(|&i| self.arguments[i]).collect();
        let mut input_parameters = vec![None; self.input_count];
        for (parameter, argument) in arguments.iter().enumerate() {
            if let Argument::Input(index) = argument {
                input_parameters[*index] = Some(parameter);
            }
        }
        Ok(StateProgram(Rc::new(Plan {
            executable,
            owner: self.owner.clone(),
            arguments,
            input_parameters,
            input_count: self.input_count,
            visible_count: outputs.len(),
            types: all_outputs.iter().map(Tensor::ty).collect(),
        })))
    }
}
// Visible inputs are registered in input-index order, interleaved with state.
// Build the reverse mapping once per shared plan, not once per weight binding.
fn input_parameters(arguments: &[Argument]) -> Vec<usize> {
    let mut parameters = Vec::new();
    for (parameter, argument) in arguments.iter().enumerate() {
        if let Argument::Input(index) = argument {
            debug_assert_eq!(*index, parameters.len());
            parameters.push(parameter);
        }
    }
    parameters
}
fn validate_slot(owner: &Arc<()>, count: usize, slot: &StateSlot) -> Result<()> {
    if !Arc::ptr_eq(owner, &slot.owner) || slot.index >= count {
        return Err(err("state slot belongs to another schema"));
    }
    Ok(())
}
fn schema_layout(
    owner: &Arc<()>,
    types: &[TensorType],
    slots: &[StateSlot],
) -> Result<Vec<(DType, Vec<i64>)>> {
    if slots.len() != types.len() {
        return Err(err("state layout requires every slot exactly once"));
    }
    let mut seen = vec![false; types.len()];
    let mut layout = Vec::with_capacity(types.len());
    for slot in slots {
        validate_slot(owner, types.len(), slot)?;
        if std::mem::replace(&mut seen[slot.index], true) {
            return Err(err("duplicate state layout slot"));
        }
        let ty = &types[slot.index];
        layout.push((ty.dtype, ty.dims.clone()));
    }
    Ok(layout)
}
fn validate_buffer(client: &Client, buffer: &Buffer, ty: &TensorType) -> Result<()> {
    if !buffer.belongs_to(client) {
        return Err(err("state/result buffer belongs to another client"));
    }
    if buffer.dtype()? != ty.dtype || buffer.dimensions()? != ty.dims {
        return Err(err("state/result buffer shape or dtype mismatch"));
    }
    Ok(())
}
struct Plan {
    executable: Arc<Executable>,
    owner: Arc<()>,
    arguments: Vec<Argument>,
    input_parameters: Vec<Option<usize>>,
    input_count: usize,
    visible_count: usize,
    types: Vec<TensorType>,
}

/// Immutable lowered transition plus state schema and argument mappings.
/// Contains no device buffers, executable or source graph, so it can be shared
/// across threads. Compilation creates a client-local plan; each Session must
/// still initialize/bind its own buffers. This is not a runtime-state checkpoint.
pub struct PreparedStateGraph {
    lowered: LoweredProgram,
    owner: Arc<()>,
    arguments: Vec<Argument>,
    input_parameters: Vec<Option<usize>>,
    input_count: usize,
    visible_count: usize,
    types: Vec<TensorType>,
}
impl PreparedStateGraph {
    /// Inspect a visible result before compilation. Hidden state update roots
    /// are excluded; inspect those with state_type/state_layout instead.
    pub fn output_spec(&self, index: usize) -> Option<OutputSpec<'_>> {
        if index >= self.visible_count {
            return None;
        }
        let ty = &self.types[index];
        Some(OutputSpec {
            shape: &ty.dims,
            dtype: ty.dtype,
        })
    }

    /// Inspect an input-backed or resident parameter before native compilation.
    /// Returns storage dtype (including BF16), not a converted symbolic dtype.
    /// Foreign, pruned and later-registered parameters are rejected.
    pub fn parameter_type(&self, parameter: &Parameter) -> Result<(DType, Vec<i64>)> {
        if !Arc::ptr_eq(&parameter.owner, &self.owner) {
            return Err(err("parameter belongs to another schema"));
        }
        match &parameter.storage {
            ParameterStorage::State(slot) => self.state_type(slot),
            ParameterStorage::Input(index) => {
                let index = self
                    .input_parameters
                    .get(*index)
                    .copied()
                    .flatten()
                    .ok_or_else(|| err("parameter is absent or pruned from this program"))?;
                let spec = self
                    .lowered
                    .input_spec(index)
                    .expect("prepared parameter mapping");
                Ok((spec.dtype, spec.shape.to_vec()))
            }
        }
    }

    /// Original visible input indices in runtime order, without loading a plugin.
    pub fn input_indices(&self) -> Vec<usize> {
        self.input_parameters
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|_| i))
            .collect()
    }

    /// Inspect a retained state identity before compilation. Foreign and later
    /// registered slots fail; returned shapes are owned host metadata.
    pub fn state_type(&self, slot: &StateSlot) -> Result<(DType, Vec<i64>)> {
        validate_slot(&self.owner, self.types.len() - self.visible_count, slot)?;
        let ty = &self.types[self.visible_count + slot.index];
        Ok((ty.dtype, ty.dims.clone()))
    }

    /// Validate a complete unique slot set, returning types in caller order.
    /// This performs no compilation, device allocation or runtime state reads.
    pub fn state_layout(&self, slots: &[StateSlot]) -> Result<Vec<(DType, Vec<i64>)>> {
        schema_layout(&self.owner, &self.types[self.visible_count..], slots)
    }

    /// Compile or restore this transition through the supplied compiler's cache,
    /// without repeating graph lowering/encoding. Reuses executable cache keys
    /// with ordinary state compilation, but retains this snapshot's slot schema.
    pub fn compile(&self, compiler: &mut Compiler) -> Result<StateProgram> {
        Ok(StateProgram(Rc::new(Plan {
            executable: compiler.compile_lowered(&self.lowered)?,
            owner: self.owner.clone(),
            arguments: self.arguments.clone(),
            input_parameters: self.input_parameters.clone(),
            input_count: self.input_count,
            visible_count: self.visible_count,
            types: self.types.clone(),
        })))
    }
}
/// A compiled state transition; clones share code, never session state.
#[derive(Clone)]
pub struct StateProgram(Rc<Plan>);
impl StateProgram {
    /// Original visible-input registration numbers required by this plan, in
    /// runtime order before fixed binding. Hidden state inputs are not listed.
    pub fn input_indices(&self) -> Vec<usize> {
        self.0
            .input_parameters
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|_| i))
            .collect()
    }
    /// Client owning this compiled plan and its state buffers.
    pub fn client(&self) -> &Client {
        &self.0.executable.client
    }

    /// Inspect one compiled state slot without requiring the full state set.
    /// Foreign slots and slots registered after this plan was compiled fail.
    /// No device allocation or execution occurs; state_layout still requires
    /// a complete, unique set for checkpoint validation.
    pub fn state_type(&self, slot: &StateSlot) -> Result<(DType, Vec<i64>)> {
        let count = self.0.types.len() - self.0.visible_count;
        validate_slot(&self.0.owner, count, slot)?;
        let ty = &self.0.types[self.0.visible_count + slot.index];
        Ok((ty.dtype, ty.dims.clone()))
    }

    /// Allocate a zero-filled buffer for one state slot.
    ///
    /// Unlike [`Self::zero_state`], this does not require a complete state
    /// layout. It is intended for builders that already have explicit buffers
    /// for some slots and need defaults only for the remainder.
    pub fn zero_state_slot(&self, slot: &StateSlot) -> Result<Buffer> {
        let (dtype, shape) = self.state_type(slot)?;
        self.zero_buffer(dtype, &shape)
    }

    /// Allocate independent zero-filled buffers for every state slot.
    ///
    /// Slots may be in any order, but must be complete, unique and belong to
    /// this program. The entire layout is validated before any upload. Returned
    /// buffers can initialize `session` or `Session::new_session`; this does not
    /// compile, execute, copy existing state or bind fixed inputs.
    ///
    /// This explicitly zeros *all* state, including resident model parameters:
    /// use checkpoint initialization instead when zeros are not appropriate.
    /// Allocation uses host staging and device uploads, not a device memset.
    /// On a returned upload error, already-created buffers are dropped. Host
    /// staging uses ordinary Rust allocation and may abort on out-of-memory.
    pub fn zero_state(&self, slots: &[StateSlot]) -> Result<Vec<(StateSlot, Buffer)>> {
        let layout = self.state_layout(slots)?;
        slots
            .iter()
            .zip(layout)
            .map(|(slot, (dtype, shape))| Ok((slot.clone(), self.zero_buffer(dtype, &shape)?)))
            .collect()
    }

    fn zero_buffer(&self, dtype: DType, shape: &[i64]) -> Result<Buffer> {
        let len = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
            .ok_or_else(|| err("state element count overflow"))?;
        match dtype {
            DType::F32 => Ok(self.client().buffer(shape, &vec![0.; len])?),
            DType::I32 => Ok(self.client().buffer(shape, &vec![0; len])?),
            DType::BF16 => Ok(self.client().buffer_bf16_bits(shape, &vec![0; len])?),
            DType::U8 => Ok(self.client().buffer(shape, &vec![0u8; len])?),
            dtype => Err(err(format!("cannot initialize state dtype {dtype:?}"))),
        }
    }

    /// Inspect a retained input-backed or resident parameter in this plan.
    /// Rejects foreign, pruned and later-registered handles before any upload.
    pub fn parameter_type(&self, parameter: &Parameter) -> Result<(DType, Vec<i64>)> {
        if !Arc::ptr_eq(&parameter.owner, &self.0.owner) {
            return Err(err("parameter belongs to another schema"));
        }
        match &parameter.storage {
            ParameterStorage::State(slot) => self.state_type(slot),
            ParameterStorage::Input(index) => {
                let index = self
                    .0
                    .input_parameters
                    .get(*index)
                    .copied()
                    .flatten()
                    .ok_or_else(|| err("parameter is absent or pruned from this program"))?;
                let ty = &self.0.executable.inputs[index];
                Ok((ty.dtype, ty.dims.clone()))
            }
        }
    }

    /// Validate a complete, arbitrarily ordered set of state identities and
    /// return (dtype, shape) in that order, without allocating device buffers.
    /// Missing, duplicate and foreign slots fail. This supports checkpoint
    /// preflight without exposing process-local slot indices as persistent IDs.
    pub fn state_layout(&self, slots: &[StateSlot]) -> Result<Vec<(DType, Vec<i64>)>> {
        schema_layout(&self.0.owner, &self.0.types[self.0.visible_count..], slots)
    }

    /// Build a complete source -> destination mapping by explicit, unique names.
    /// Both lists must cover their program's entire state exactly once and use
    /// the same nonempty name set. Matched slots must have identical dtype/shape.
    /// List order and graph-local identities may differ; output follows source
    /// list order. Names are application-owned, not inferred from modules, and
    /// are compared exactly (no aliases, normalization or partial matching).
    ///
    /// This preflight does not inspect buffers, compile, execute or change a
    /// session. Reuse the result with Session::switch_program, which additionally
    /// validates buffer/client compatibility and replacement input bindings.
    /// Names must still have the same semantic meaning in both model versions;
    /// equal names and shapes do not prove that a model migration is correct.
    pub fn map_state_by_name(
        &self,
        source: &[(&str, StateSlot)],
        destination: &StateProgram,
        target: &[(&str, StateSlot)],
    ) -> Result<Vec<(StateSlot, StateSlot)>> {
        let source_layout = self.state_layout(
            &source
                .iter()
                .map(|(_, slot)| slot.clone())
                .collect::<Vec<_>>(),
        )?;
        let target_layout = destination.state_layout(
            &target
                .iter()
                .map(|(_, slot)| slot.clone())
                .collect::<Vec<_>>(),
        )?;
        if source.len() != target.len() {
            return Err(err("state mapping name sets differ"));
        }
        let mut by_name = std::collections::HashMap::with_capacity(target.len());
        for (index, (name, _)) in target.iter().enumerate() {
            if name.is_empty() || by_name.insert(*name, index).is_some() {
                return Err(err("destination state names must be nonempty and unique"));
            }
        }
        let mut seen = std::collections::HashSet::with_capacity(source.len());
        let mut mapping = Vec::with_capacity(source.len());
        for ((name, slot), layout) in source.iter().zip(source_layout) {
            if name.is_empty() || !seen.insert(*name) {
                return Err(err("source state names must be nonempty and unique"));
            }
            let index = *by_name
                .get(name)
                .ok_or_else(|| err(format!("destination state name missing: {name}")))?;
            if layout != target_layout[index] {
                return Err(err(format!(
                    "state mapping dtype or shape mismatch: {name}"
                )));
            }
            mapping.push((slot.clone(), target[index].1.clone()));
        }
        Ok(mapping)
    }

    /// Bind initial state by slot identity. Order is irrelevant; every slot must
    /// appear exactly once. No compilation occurs when creating another session.
    pub fn session(&self, initial: Vec<(StateSlot, Buffer)>) -> Result<Session> {
        let mut dynamic_input_count = 0;
        let inputs = self
            .0
            .input_parameters
            .iter()
            .map(|parameter| {
                if parameter.is_some() {
                    let index = dynamic_input_count;
                    dynamic_input_count += 1;
                    InputBinding::Dynamic(index)
                } else {
                    InputBinding::Unused
                }
            })
            .collect();
        Ok(Session {
            program: self.clone(),
            states: self.bind_states(initial)?,
            inputs,
            dynamic_input_count,
        })
    }
    fn bind_states(&self, initial: Vec<(StateSlot, Buffer)>) -> Result<Vec<Buffer>> {
        let count = self.0.types.len() - self.0.visible_count;
        if initial.len() != count {
            return Err(err("initial state count mismatch"));
        }
        let mut states: Vec<Option<Buffer>> = (0..count).map(|_| None).collect();
        for (slot, buffer) in initial {
            validate_slot(&self.0.owner, count, &slot)?;
            validate_buffer(
                &self.0.executable.client,
                &buffer,
                &self.0.types[self.0.visible_count + slot.index],
            )?;
            if states[slot.index].is_some() {
                return Err(err("duplicate initial state slot"));
            }
            states[slot.index] = Some(buffer);
        }
        Ok(states.into_iter().map(Option::unwrap).collect())
    }
    fn identify_states(&self, states: Vec<Buffer>) -> Vec<(StateSlot, Buffer)> {
        states
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| {
                (
                    StateSlot {
                        owner: self.0.owner.clone(),
                        index,
                    },
                    buffer,
                )
            })
            .collect()
    }
}

/// Exclusive, synchronous session state. There is no Clone or interior mutation.
pub struct Session {
    program: StateProgram,
    states: Vec<Buffer>,
    inputs: Vec<InputBinding>,
    dynamic_input_count: usize,
}

fn input_parameter_bindings(
    program: &StateProgram,
    bindings: Vec<(Parameter, Rc<Buffer>)>,
) -> Result<Vec<(usize, Rc<Buffer>)>> {
    let mut indexed = Vec::with_capacity(bindings.len());
    for (parameter, buffer) in bindings {
        if !Arc::ptr_eq(&parameter.owner, &program.0.owner) {
            return Err(err("parameter belongs to another schema"));
        }
        let ParameterStorage::Input(index) = parameter.storage else {
            return Err(err(
                "resident parameters require state initialization, not fixed input binding",
            ));
        };
        indexed.push((index, buffer));
    }
    Ok(indexed)
}
impl Session {
    /// Switch plans using destination parameter identities for fixed inputs.
    /// Same complete state mapping, zero-copy movement and error atomicity as
    /// switch_program, without exposing numeric input registration indices.
    /// Foreign, state-backed, duplicate and pruned parameters are rejected;
    /// bindings must refer to the destination plan, not the previous graph.
    pub fn switch_program_parameters(
        &mut self,
        program: &StateProgram,
        mapping: &[(StateSlot, StateSlot)],
        bindings: Vec<(Parameter, Rc<Buffer>)>,
    ) -> Result<()> {
        let indexed = input_parameter_bindings(program, bindings)?;
        self.switch_program(program, mapping, indexed)
    }
    /// Switch to a compiled plan using a complete source -> destination state
    /// mapping and an explicit replacement set of fixed visible-input bindings.
    /// Every source and destination slot must occur exactly once, with matching
    /// dtype/shape; all buffers must belong to the destination client. Mapping
    /// order is irrelevant. Equal-shaped but semantically wrong mappings cannot
    /// be detected. Different state counts require application-side migration.
    ///
    /// State buffers move without execution, compilation, upload, download or
    /// payload copying. Fixed bindings are NOT inferred from the previous plan:
    /// pass their destination input numbers and shared buffers explicitly (empty
    /// clears them). Unbound retained inputs become dynamic in destination order.
    /// On any validation error, the old plan, bindings and state remain intact;
    /// supplied fixed-buffer handles are consumed. No Send/Sync capability is added.
    pub fn switch_program(
        &mut self,
        program: &StateProgram,
        mapping: &[(StateSlot, StateSlot)],
        fixed_inputs: Vec<(usize, Rc<Buffer>)>,
    ) -> Result<()> {
        let source_slots: Vec<_> = mapping.iter().map(|(source, _)| source.clone()).collect();
        let destination_slots: Vec<_> = mapping
            .iter()
            .map(|(_, destination)| destination.clone())
            .collect();
        let source_layout = self.program.state_layout(&source_slots)?;
        let destination_layout = program.state_layout(&destination_slots)?;
        if source_layout != destination_layout {
            return Err(err("program switch state dtype or shape mismatch"));
        }
        let mut destination_indices = vec![0; self.states.len()];
        for (source, destination) in mapping {
            validate_buffer(
                program.client(),
                &self.states[source.index],
                &program.0.types[program.0.visible_count + destination.index],
            )?;
            destination_indices[source.index] = destination.index;
        }
        // Stage binding validation before moving any live state. This candidate
        // is private and cannot execute until all state buffers are installed.
        let mut candidate = Self {
            program: program.clone(),
            states: Vec::new(),
            inputs: Vec::new(),
            dynamic_input_count: 0,
        };
        candidate.bind_inputs(fixed_inputs)?;
        let mut states: Vec<Option<Buffer>> = (0..self.states.len()).map(|_| None).collect();
        for (source, buffer) in std::mem::take(&mut self.states).into_iter().enumerate() {
            states[destination_indices[source]] = Some(buffer);
        }
        candidate.states = states.into_iter().map(Option::unwrap).collect();
        *self = candidate;
        Ok(())
    }

    /// Create an independent session with supplied state, inheriting this
    /// session's compiled program and current fixed input/parameter bindings.
    /// Every state slot must be supplied exactly once, just like
    /// `StateProgram::session`; invalid state leaves this session unchanged.
    /// Supplied buffers are consumed, including on error.
    ///
    /// Fixed buffers are shared read-only via Rc; later rebinding either session
    /// does not affect the other. Dynamic inputs keep their existing order.
    /// This does not copy the current state, initialize missing slots, compile,
    /// execute, or transfer tensor payloads. It is not a snapshot/fork or a
    /// cross-thread spawning API; sessions remain exclusive and synchronous.
    pub fn new_session(&self, initial: Vec<(StateSlot, Buffer)>) -> Result<Self> {
        let states = self.program.bind_states(initial)?;
        Ok(Self {
            program: self.program.clone(),
            states,
            inputs: self.inputs.clone(),
            dynamic_input_count: self.dynamic_input_count,
        })
    }

    /// Compiled transition and state schema used by this session. Sharing this
    /// plan does not share or clone the session's mutable buffers.
    pub fn program(&self) -> &StateProgram {
        &self.program
    }
    /// Borrow the current resident buffer for a parameter handle. State-backed
    /// parameters reflect completed updates/replacements; input parameters must
    /// have a fixed binding (by identity or numeric input index).
    ///
    /// Dynamic call arguments are not retained and cannot be inspected here.
    /// Foreign handles and parameters added after this program was compiled are
    /// rejected. This performs no native calls, uploads, downloads or copies.
    /// The borrow prevents mutating this session until the buffer borrow ends;
    /// it is not a snapshot or an owned/shared mutable parameter handle.
    pub fn parameter(&self, parameter: &Parameter) -> Result<&Buffer> {
        if !Arc::ptr_eq(&parameter.owner, &self.program.0.owner) {
            return Err(err("parameter belongs to another schema"));
        }
        match &parameter.storage {
            ParameterStorage::State(slot) => self.state(slot),
            ParameterStorage::Input(index) => match self.inputs.get(*index) {
                Some(InputBinding::Fixed(buffer)) => Ok(buffer.as_ref()),
                Some(InputBinding::Dynamic(_)) => {
                    Err(err("parameter has no fixed resident binding"))
                }
                Some(InputBinding::Unused) => Err(err("parameter was pruned from this program")),
                None => Err(err("parameter is not part of this compiled program")),
            },
        }
    }

    /// Replace fixed inputs by parameter identity, without numeric input indices.
    /// Order is irrelevant. Foreign or duplicate handles, including clones of
    /// the same parameter, are rejected before any bindings/state are replaced.
    /// This replaces the entire fixed set, including prior `bind_inputs` calls;
    /// empty clears it. Unbound inputs retain registration order for `run`.
    /// Parameters removed by `compile_pruned` cannot be bound.
    /// Parameter handles need not remain alive after binding. Buffers are shared
    /// read-only via Rc; no compilation, execution, or tensor copies occur.
    pub fn bind_parameters(&mut self, bindings: Vec<(Parameter, Rc<Buffer>)>) -> Result<()> {
        self.bind_inputs(input_parameter_bindings(&self.program, bindings)?)
    }
    /// Replace the set of fixed visible inputs (e.g. resident inference weights).
    /// Indices refer to visible input registration order, excluding state slots.
    /// `run` then takes only unbound inputs, in their original relative order.
    /// For pruned plans, indices are still original registration numbers;
    /// removed inputs cannot be bound and never appear in `run` arguments.
    /// An empty set restores the original calling convention.
    ///
    /// Buffers are shared by ownership, not copied or embedded as graph constants.
    /// All bindings are validated before replacement; errors leave both previous
    /// bindings and mutable state intact. No compilation or execution occurs.
    pub fn bind_inputs(&mut self, bindings: Vec<(usize, Rc<Buffer>)>) -> Result<()> {
        let plan = &self.program.0;
        let mut fixed: Vec<Option<Rc<Buffer>>> = (0..plan.input_count).map(|_| None).collect();
        for (index, buffer) in bindings {
            let slot = fixed
                .get_mut(index)
                .ok_or_else(|| err("bound input index out of range"))?;
            if slot.is_some() {
                return Err(err("duplicate bound input index"));
            }
            let parameter =
                plan.input_parameters[index].ok_or_else(|| err("cannot bind a pruned input"))?;
            let expected = &plan.executable.inputs[parameter];
            if !buffer.belongs_to(&plan.executable.client) {
                return Err(err("bound input belongs to another client"));
            }
            if buffer.dtype()? != expected.dtype || buffer.dimensions()? != expected.dims {
                return Err(err(format!("bound input {index}: shape or dtype mismatch")));
            }
            *slot = Some(buffer);
        }
        let mut dynamic_count = 0;
        let inputs = fixed
            .into_iter()
            .enumerate()
            .map(|(original, buffer)| match buffer {
                Some(buffer) => InputBinding::Fixed(buffer),
                None if plan.input_parameters[original].is_none() => InputBinding::Unused,
                None => {
                    let index = dynamic_count;
                    dynamic_count += 1;
                    InputBinding::Dynamic(index)
                }
            })
            .collect();
        self.inputs = inputs;
        self.dynamic_input_count = dynamic_count;
        Ok(())
    }

    /// Number of buffers required by `run` after applying fixed input bindings.
    pub fn input_count(&self) -> usize {
        self.dynamic_input_count
    }
    /// Replace the complete resident state without compiling or executing a graph.
    /// Bindings may be in any order; every slot must occur exactly once. All are
    /// validated before commit, so an error leaves this session unchanged.
    /// Supplied buffers are consumed (and dropped on error).
    ///
    /// Returns the previous buffers with their slot identities, suitable for a
    /// later replacement or another session of this program. This transfers
    /// ownership, not a deep snapshot: no device buffers are copied or downloaded.
    pub fn replace_state(
        &mut self,
        replacement: Vec<(StateSlot, Buffer)>,
    ) -> Result<Vec<(StateSlot, Buffer)>> {
        let states = self.program.bind_states(replacement)?;
        Ok(self
            .program
            .identify_states(std::mem::replace(&mut self.states, states)))
    }
    /// Consume this session and transfer all resident state, in registration order.
    /// No replacement buffers, graph execution, tensor copies, or downloads are
    /// needed. The result can initialize another session of the same schema/client
    /// or be passed to `replace_state`; it is not a serialized checkpoint.
    ///
    /// Fixed input/parameter bindings are dropped, not included in the result.
    /// Retain the program and any shared weights needed to resume, and rebind
    /// those weights on the new session. Device buffers remain owned and live
    /// while held here; this does not offload them or make them Send/Sync.
    pub fn into_state(self) -> Vec<(StateSlot, Buffer)> {
        self.program.identify_states(self.states)
    }
    /// Deep-copy every resident state into another client's selected device.
    /// Copies synchronously via host memory, in registration order, preserving
    /// slot identities. The result can initialize a separately compiled program
    /// from the same state schema; fixed input/weight bindings are not included.
    ///
    /// `max_tensor_host_bytes` bounds each tensor's host payload, not total state
    /// or device/allocator memory. Even the same client receives fresh buffers.
    /// On failure, partial destination copies are dropped and this session is
    /// unchanged. This is a quiescent snapshot, not live migration or zero-copy
    /// transport; the returned buffers remain thread-affine.
    pub fn copy_state_to_client_via_host(
        &self,
        destination: &Client,
        max_tensor_host_bytes: usize,
    ) -> Result<Vec<(StateSlot, Buffer)>> {
        self.copy_state_to_client_via_host_with_limits(
            destination,
            max_tensor_host_bytes,
            usize::MAX,
        )
    }
    /// Copy state with both per-tensor and total host-payload limits.
    /// All native payload sizes are checked before the first payload download
    /// or destination upload. Total bytes count every slot, including aliases.
    /// Copies remain sequential: this bounds transfer volume, not peak memory,
    /// allocator overhead, concurrent sessions, or a device-memory reservation.
    /// Fixed bindings are excluded. Failure leaves the source unchanged.
    pub fn copy_state_to_client_via_host_with_limits(
        &self,
        destination: &Client,
        max_tensor_host_bytes: usize,
        max_total_host_bytes: usize,
    ) -> Result<Vec<(StateSlot, Buffer)>> {
        let mut total = 0usize;
        for buffer in &self.states {
            let bytes = buffer.host_payload_bytes()?;
            if bytes > max_tensor_host_bytes {
                return Err(err("state tensor host payload exceeds per-tensor limit"));
            }
            total = total
                .checked_add(bytes)
                .ok_or_else(|| err("state host payload size overflow"))?;
            if total > max_total_host_bytes {
                return Err(err("state host payload exceeds total limit"));
            }
        }
        let copies =
            self.states
                .iter()
                .map(|buffer| {
                    Ok(buffer
                        .copy_to_client_via_host_with_limit(destination, max_tensor_host_bytes)?)
                })
                .collect::<Result<Vec<_>>>()?;
        Ok(self.program.identify_states(copies))
    }
    pub fn state(&self, slot: &StateSlot) -> Result<&Buffer> {
        validate_slot(&self.program.0.owner, self.states.len(), slot)?;
        Ok(&self.states[slot.index])
    }
    pub fn run(&mut self, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        let _span = tracing::debug_span!(
            "xla.session.run",
            dynamic_inputs = inputs.len(),
            states = self.states.len()
        )
        .entered();
        let executable = self.program.0.executable.clone();
        self.run_with(inputs, |arguments| executable.execute(arguments))
    }
    fn run_with(
        &mut self,
        inputs: &[&Buffer],
        execute: impl FnOnce(&[&Buffer]) -> Result<Vec<Buffer>>,
    ) -> Result<Vec<Buffer>> {
        let plan = &self.program.0;
        if inputs.len() != self.dynamic_input_count {
            return Err(err("session input count mismatch"));
        }
        let arguments: Vec<_> = plan
            .arguments
            .iter()
            .map(|arg| match *arg {
                Argument::Input(i) => match &self.inputs[i] {
                    InputBinding::Dynamic(index) => inputs[*index],
                    InputBinding::Fixed(buffer) => buffer.as_ref(),
                    InputBinding::Unused => {
                        unreachable!("pruned inputs have no execution argument")
                    }
                },
                Argument::State(i) => &self.states[i],
            })
            .collect();
        let mut outputs = execute(&arguments)?;
        if outputs.len() != plan.types.len() {
            return Err(err("stateful execution output count mismatch"));
        }
        for (buffer, ty) in outputs.iter().zip(&plan.types) {
            validate_buffer(&plan.executable.client, buffer, ty)?;
        }
        // No slot changes until all outputs have completed and passed validation.
        self.states = outputs.split_off(plan.visible_count);
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_training_graph_is_built_in_pliron() {
        let mut graph = StateGraph::default();
        let input = graph.input(&[2, 3]).unwrap();
        let weight = graph.trainable_parameter(&[3, 1]).unwrap();
        let loss = input
            .matmul(weight.tensor())
            .unwrap()
            .sum(&[0, 1], false)
            .unwrap();
        let gradient = loss.grad(&[weight.tensor().clone()]).unwrap().remove(0);
        assert!(
            graph
                .graph
                .stablehlo(&gradient)
                .unwrap()
                .contains("stablehlo.dot_general")
        );
    }

    #[test]
    fn pruned_state_abi_uses_pliron_parameter_numbers() {
        let mut graph = StateGraph::default();
        let _unused = graph.parameter(&[2]).unwrap();
        let used = graph.parameter(&[2]).unwrap();
        let output = used.tensor().add(used.tensor()).unwrap();
        let (lowered, parameters) = graph.graph.prepare_pruned(&[output]).unwrap();
        assert_eq!(lowered.format(), "mlir");
        assert_eq!(parameters, [1]);
        assert_eq!(lowered.input_count(), 1);
    }

    #[test]
    fn conditional_update_validation_preserves_all_versions() {
        let mut g = StateGraph::default();
        let a = g.state(&[2]).unwrap();
        let b = g.state_i32(&[]).unwrap();
        let old_a = g.read(&a).unwrap();
        let old_b = g.read(&b).unwrap();
        let yes = g.constant(&[], &[1.]).unwrap();
        let good = g.constant(&[2], &[5., 6.]).unwrap();
        let wrong = g.constant(&[], &[1.]).unwrap();
        let foreign = Graph::default().input(&[]).unwrap();
        let vector = g.constant(&[1], &[1.]).unwrap();
        for updates in [
            vec![(&a, good.clone()), (&b, wrong)],
            vec![(&a, good.clone()), (&a, good.clone())],
        ] {
            assert!(g.write_many_if(&yes, &updates).is_err());
            assert_eq!(g.read(&a).unwrap().node_id(), old_a.node_id());
            assert_eq!(g.read(&b).unwrap().node_id(), old_b.node_id());
        }
        for condition in [foreign, vector] {
            assert!(g.write_many_if(&condition, &[(&a, good.clone())]).is_err());
            assert_eq!(g.read(&a).unwrap().node_id(), old_a.node_id());
        }
        g.write_many_if(&yes, &[]).unwrap();
        assert_eq!(g.read(&a).unwrap().node_id(), old_a.node_id());
        assert_eq!(g.read(&b).unwrap().node_id(), old_b.node_id());
    }
    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn invalid_visible_integer_result_prevents_hidden_state_commit() {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let mut graph = StateGraph::default();
        let state = graph.state_i32(&[]).unwrap();
        let value = graph.read(&state).unwrap();
        let program = graph.compile(&mut compiler, &[value]).unwrap();
        let mut session = program
            .session(vec![(state.clone(), client.buffer(&[], &[7]).unwrap())])
            .unwrap();
        assert!(
            session
                .run_with(&[], |_| Ok(vec![
                    client.buffer(&[], &[99.])?, // Invalid visible dtype.
                    client.buffer(&[], &[99])?,  // Otherwise valid hidden state.
                ]))
                .is_err()
        );
        assert_eq!(session.state(&state).unwrap().to_vec::<i32>().unwrap(), [7]);
    }
    #[test]
    fn mixed_symbolic_write_failure_preserves_versions() {
        let mut g = StateGraph::default();
        let f = g.state(&[]).unwrap();
        let i = g.state_i32(&[]).unwrap();
        let old_f = g.read(&f).unwrap();
        let old_i = g.read(&i).unwrap();
        let next_f = old_f.add_scalar(1.).unwrap();
        let next_i = old_i.wrapping_add_scalar(1).unwrap();
        for updates in [
            vec![(&f, next_f.clone()), (&i, next_f.clone())],
            vec![(&i, next_i.clone()), (&i, next_i.clone())],
        ] {
            assert!(g.record_updates(&updates).is_err());
            assert_eq!(g.read(&f).unwrap().node_id(), old_f.node_id());
            assert_eq!(g.read(&i).unwrap().node_id(), old_i.node_id());
        }
    }
    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn wrong_integer_result_does_not_commit_any_state() {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let mut g = StateGraph::default();
        let f = g.state(&[]).unwrap();
        let i = g.state_i32(&[]).unwrap();
        let program = g.compile(&mut compiler, &[]).unwrap();
        let mut session = program
            .session(vec![
                (f.clone(), client.buffer(&[], &[1.]).unwrap()),
                (i.clone(), client.buffer(&[], &[2]).unwrap()),
            ])
            .unwrap();
        assert!(
            session
                .run_with(&[], |_| Ok(vec![
                    client.buffer(&[], &[99.])?,
                    client.buffer(&[], &[99.])?
                ]))
                .is_err()
        );
        assert_eq!(session.state(&f).unwrap().to_vec::<f32>().unwrap(), [1.]);
        assert_eq!(session.state(&i).unwrap().to_vec::<i32>().unwrap(), [2]);
    }
    #[test]
    fn visible_parameter_map_preserves_interleaved_registration() {
        assert!(input_parameters(&[]).is_empty());
        assert!(input_parameters(&[Argument::State(0)]).is_empty());
        let arguments: Vec<_> = (0..1024)
            .flat_map(|i| [Argument::State(i), Argument::Input(i)])
            .collect();
        assert_eq!(
            input_parameters(&arguments),
            (0..1024).map(|i| 2 * i + 1).collect::<Vec<_>>()
        );
    }
    #[test]
    fn batch_symbolic_writes_validate_before_commit() {
        let mut graph = StateGraph::default();
        let a = graph.state(&[]).unwrap();
        let b = graph.state(&[]).unwrap();
        let old_a = graph.read(&a).unwrap();
        let old_b = graph.read(&b).unwrap();
        let next = old_a.add_scalar(1.).unwrap();
        let wrong_shape = graph.constant(&[1], &[1.]).unwrap();
        let mut foreign = StateGraph::default();
        let foreign_slot = foreign.state(&[]).unwrap();
        let foreign_value = foreign.read(&foreign_slot).unwrap();
        let duplicate = a.clone();
        for updates in [
            [(&a, &next), (&b, &wrong_shape)],
            [(&a, &next), (&b, &foreign_value)],
            [(&a, &next), (&foreign_slot, &next)],
            [(&a, &next), (&duplicate, &old_a)],
        ] {
            assert!(graph.write_many(&updates).is_err());
            assert_eq!(graph.read(&a).unwrap().node_id(), old_a.node_id());
            assert_eq!(graph.read(&b).unwrap().node_id(), old_b.node_id());
        }
        graph.write_many(&[]).unwrap();
        assert_eq!(graph.read(&a).unwrap().node_id(), old_a.node_id());
        graph.write_many(&[(&a, &next)]).unwrap();
        let written_a = graph.read(&a).unwrap();
        assert_ne!(written_a.node_id(), old_a.node_id());
        assert_eq!(graph.read(&b).unwrap().node_id(), old_b.node_id());
        graph.write_many(&[(&b, &next), (&a, &old_b)]).unwrap();
        assert_ne!(graph.read(&a).unwrap().node_id(), written_a.node_id());
        assert_ne!(graph.read(&b).unwrap().node_id(), old_b.node_id());
    }
    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn rejected_execution_and_partial_results_do_not_commit() {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let mut graph = StateGraph::default();
        let a = graph.state(&[]).unwrap();
        let b = graph.state(&[]).unwrap();
        let program = graph.compile(&mut compiler, &[]).unwrap();
        let mut session = program
            .session(vec![
                (a.clone(), client.buffer(&[], &[1.]).unwrap()),
                (b.clone(), client.buffer(&[], &[2.]).unwrap()),
            ])
            .unwrap();
        // Inject errors at the execution boundary; do not manufacture a crashing
        // or malicious native plugin to test session commit semantics.
        assert!(
            session
                .run_with(&[], |_| Err(err("injected execution failure")))
                .is_err()
        );
        assert!(session.run_with(&[], |_| Ok(vec![])).is_err());
        assert!(
            session
                .run_with(&[], |_| Ok(vec![
                    client.buffer(&[], &[100.])?,
                    client.buffer(&[1], &[200.])?,
                ]))
                .is_err()
        );
        assert_eq!(session.state(&a).unwrap().to_vec::<f32>().unwrap(), [1.]);
        assert_eq!(session.state(&b).unwrap().to_vec::<f32>().unwrap(), [2.]);
    }
}
