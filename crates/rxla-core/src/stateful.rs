//! Lazy-tensor-facing state effects with an explicit single-use step boundary.

use super::*;
use std::collections::BTreeMap;

struct InputTrace {
    nodes: Vec<(Op, Vec<usize>, TensorType)>,
    roots: Vec<usize>,
    bindings: Vec<Tensor>,
    fingerprint: String,
}

impl InputTrace {
    fn capture(inputs: &[Tensor]) -> Result<Self> {
        if inputs.is_empty() {
            return Ok(Self {
                nodes: Vec::new(),
                roots: Vec::new(),
                bindings: Vec::new(),
                fingerprint: "empty".to_owned(),
            });
        }
        let graph = inputs[0].graph();
        if inputs
            .iter()
            .any(|input| !Arc::ptr_eq(&graph.0, &input.graph().0))
        {
            return Err(err("stateful inputs belong to different lazy sessions"));
        }
        if inputs.iter().any(|input| !input.is_implicit_lazy()) {
            return Err(err(
                "stateful input fusion requires implicit lazy tensors, not explicit Tracer values",
            ));
        }
        let source = graph.0.lock().map_err(|_| err("graph lock poisoned"))?;
        let semantic = source.semantic_nodes()?;
        let source_roots = inputs.iter().map(Tensor::node_id).collect::<Vec<_>>();
        let mut reachable = vec![false; semantic.len()];
        let mut worklist = source_roots.clone();
        while let Some(id) = worklist.pop() {
            let visited = reachable
                .get_mut(id.index())
                .ok_or_else(|| err("stateful input has an invalid SSA value"))?;
            if *visited {
                continue;
            }
            *visited = true;
            worklist.extend(semantic[id.index()].operands.iter().copied());
        }

        let mut dense = vec![None; semantic.len()];
        let mut nodes = Vec::new();
        let mut parameters = Vec::new();
        let mut fingerprint = String::new();
        for (source_index, node) in semantic.into_iter().enumerate() {
            if !reachable[source_index] {
                continue;
            }
            let operands = node
                .operands
                .iter()
                .map(|operand| {
                    dense[operand.index()]
                        .ok_or_else(|| err("stateful input graph is not topologically ordered"))
                })
                .collect::<Result<Vec<_>>>()?;
            let op = match node.op {
                Op::Parameter(number) => {
                    let normalized = parameters.len();
                    parameters.push(number);
                    Op::Parameter(normalized)
                }
                Op::StateInput { .. } | Op::StateRead { .. } | Op::StateWrite { .. } => {
                    return Err(err(
                        "nested state effects cannot be used as stateful inputs",
                    ));
                }
                op => op,
            };
            let target_index = nodes.len();
            dense[source_index] = Some(target_index);
            fingerprint.push_str(&format!("{op:?}:{operands:?}:{:?};", node.ty));
            nodes.push((op, operands, node.ty));
        }
        let roots = source_roots
            .iter()
            .map(|root| dense[root.index()].ok_or_else(|| err("stateful input root was pruned")))
            .collect::<Result<Vec<_>>>()?;
        fingerprint.push_str(&format!("roots:{roots:?}"));
        drop(source);
        let bindings = inputs[0].lazy_inputs(&parameters)?;
        Ok(Self {
            nodes,
            roots,
            bindings,
            fingerprint,
        })
    }

    fn import(self, graph: &mut StateGraph) -> Result<(Vec<Tensor>, Vec<Tensor>, String)> {
        let mut values: Vec<Tensor> = Vec::with_capacity(self.nodes.len());
        for (op, operands, ty) in self.nodes {
            let value = match op {
                Op::Parameter(_) => graph.input_dtype(&ty)?,
                op => {
                    let operands = operands
                        .iter()
                        .map(|&index| values[index].clone())
                        .collect::<Vec<_>>();
                    graph.append_dataflow(op, &operands, &ty)?
                }
            };
            values.push(value);
        }
        let roots = self
            .roots
            .iter()
            .map(|&index| values[index].clone())
            .collect();
        Ok((roots, self.bindings, self.fingerprint))
    }
}

/// A model function whose state declarations are interpreted at first call.
pub struct StatefulModel<F> {
    build: F,
}

impl<F> StatefulModel<F> {
    pub fn new(build: F) -> Self {
        Self { build }
    }

    /// Create an independent state owner. Compilation and zero initialization
    /// are deferred until the first step is evaluated.
    pub fn session(&self) -> StatefulSession<'_, F> {
        StatefulSession {
            model: self,
            initial: BTreeMap::new(),
            compiled: None,
        }
    }
}

/// Named state-effect interpreter used only while tracing a specialization.
pub struct StateCx {
    graph: StateGraph,
    scope: Vec<String>,
    states: BTreeMap<String, StateDeclaration>,
    order: Vec<String>,
}

struct StateDeclaration {
    slot: StateSlot,
    shape: Vec<i64>,
    dtype: DType,
}

/// Stable identity of one state effect in the current model trace.
#[derive(Clone)]
pub struct StateValue {
    path: String,
    slot: StateSlot,
}

impl StateCx {
    fn new(inputs: &[Tensor]) -> Result<(Self, Vec<Tensor>, Vec<Tensor>, String)> {
        let mut graph = StateGraph::default();
        let (symbolic, bindings, fingerprint) = InputTrace::capture(inputs)?.import(&mut graph)?;
        Ok((
            Self {
                graph,
                scope: Vec::new(),
                states: BTreeMap::new(),
                order: Vec::new(),
            },
            symbolic,
            bindings,
            fingerprint,
        ))
    }

    /// Declare or read named state at the point where its shape is known.
    pub fn state(&mut self, name: &str, shape: &[i64], dtype: DType) -> Result<StateValue> {
        validate_state_name(name)?;
        let path = self.path(name);
        if let Some(existing) = self.states.get(&path) {
            if existing.shape != shape || existing.dtype != dtype {
                return Err(err(format!(
                    "state declaration {path:?} changed shape or dtype"
                )));
            }
            return Ok(StateValue {
                path,
                slot: existing.slot.clone(),
            });
        }
        let slot = match dtype {
            DType::F32 | DType::I32 => self.graph.state_named(&path, shape, dtype)?,
            _ => return Err(err(format!("state dtype {dtype:?} is unsupported"))),
        };
        self.order.push(path.clone());
        self.states.insert(
            path.clone(),
            StateDeclaration {
                slot: slot.clone(),
                shape: shape.to_vec(),
                dtype,
            },
        );
        Ok(StateValue { path, slot })
    }

    pub fn scope<T>(
        &mut self,
        name: &str,
        build: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        validate_state_name(name)?;
        self.scope.push(name.to_owned());
        let result = build(self);
        self.scope.pop();
        result
    }

    pub fn read(&self, state: &StateValue) -> Result<Tensor> {
        self.validate_state(state)?;
        self.graph.read(&state.slot)
    }

    pub fn write(&mut self, state: &StateValue, value: &Tensor) -> Result<()> {
        self.validate_state(state)?;
        self.graph.write(&state.slot, value)
    }

    /// Read the current SSA version, build its replacement, and record one
    /// explicit state write. The closure runs while tracing, not at execution.
    pub fn update(
        &mut self,
        state: &StateValue,
        update: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<()> {
        let current = self.read(state)?;
        let next = update(&current)?;
        self.write(state, &next)
    }

    pub fn write_many(&mut self, updates: &[(&StateValue, &Tensor)]) -> Result<()> {
        for (state, _) in updates {
            self.validate_state(state)?;
        }
        self.graph.write_many(
            &updates
                .iter()
                .map(|(state, value)| (&state.slot, *value))
                .collect::<Vec<_>>(),
        )
    }

    pub fn write_many_if(
        &mut self,
        condition: &Tensor,
        updates: &[(&StateValue, Tensor)],
    ) -> Result<()> {
        for (state, _) in updates {
            self.validate_state(state)?;
        }
        self.graph.write_many_if(
            condition,
            &updates
                .iter()
                .map(|(state, value)| (&state.slot, value.clone()))
                .collect::<Vec<_>>(),
        )
    }

    pub fn constant(&self, shape: &[i64], values: &[f32]) -> Result<Tensor> {
        self.graph.constant(shape, values)
    }

    pub fn constant_i32(&self, shape: &[i64], values: &[i32]) -> Result<Tensor> {
        self.graph.constant_i32(shape, values)
    }

    pub fn iota_i32(&self, shape: &[i64], axis: usize) -> Result<Tensor> {
        self.graph.iota_i32(shape, axis)
    }

    fn validate_state(&self, state: &StateValue) -> Result<()> {
        match self.states.get(&state.path) {
            Some(declaration) if declaration.slot.identity() == state.slot.identity() => Ok(()),
            _ => Err(err("state value belongs to another model trace")),
        }
    }

    fn path(&self, name: &str) -> String {
        if self.scope.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{}", self.scope.join("."), name)
        }
    }

    fn finish(self, outputs: &[Tensor]) -> Result<PendingSpecialization> {
        let slots = self
            .order
            .iter()
            .map(|name| {
                let declaration = &self.states[name];
                (name.clone(), declaration.slot.clone())
            })
            .collect();
        Ok(PendingSpecialization {
            graph: self.graph,
            outputs: outputs.to_vec(),
            slots,
        })
    }
}

impl StateValue {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn read(&self, cx: &StateCx) -> Result<Tensor> {
        cx.read(self)
    }

    pub fn write(&self, cx: &mut StateCx, value: &Tensor) -> Result<()> {
        cx.write(self, value)
    }

    /// In-place spelling for a traced state write. This advances the logical
    /// SSA version; it never mutates a shared Tensor or PJRT buffer immediately.
    pub fn copy_(&self, cx: &mut StateCx, value: &Tensor) -> Result<()> {
        cx.write(self, value)
    }

    pub fn add_(&self, cx: &mut StateCx, value: &Tensor) -> Result<()> {
        cx.update(self, |current| current.add(value))
    }

    pub fn sub_(&self, cx: &mut StateCx, value: &Tensor) -> Result<()> {
        cx.update(self, |current| current.sub(value))
    }

    pub fn mul_(&self, cx: &mut StateCx, value: &Tensor) -> Result<()> {
        cx.update(self, |current| current.mul(value))
    }
}

fn validate_state_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains('.') {
        return Err(err(format!(
            "invalid state name segment {name:?}: expected nonempty and dot-free"
        )));
    }
    Ok(())
}

struct PendingSpecialization {
    graph: StateGraph,
    outputs: Vec<Tensor>,
    slots: Vec<(String, StateSlot)>,
}

struct CompiledState {
    signature: Vec<TensorType>,
    input_fingerprint: String,
    session: Session,
    output_count: usize,
    slots: Vec<(String, StateSlot)>,
}

/// Exclusive owner of one model's resident state and compiled specialization.
pub struct StatefulSession<'model, F> {
    model: &'model StatefulModel<F>,
    initial: BTreeMap<String, Buffer>,
    compiled: Option<CompiledState>,
}

impl<F> StatefulSession<'_, F> {
    /// Override zero initialization for one named state. Names are validated
    /// against the model on first evaluation.
    pub fn initialize(mut self, name: impl Into<String>, buffer: Buffer) -> Result<Self> {
        let name = name.into();
        if name.is_empty() || self.initial.insert(name.clone(), buffer).is_some() {
            return Err(err(format!(
                "duplicate or empty initial state name {name:?}"
            )));
        }
        Ok(self)
    }

    pub fn state(&self, name: &str) -> Result<&Buffer> {
        let compiled = self
            .compiled
            .as_ref()
            .ok_or_else(|| err("state is unavailable before the first evaluated step"))?;
        let (_, slot) = compiled
            .slots
            .iter()
            .find(|(declared, _)| declared == name)
            .ok_or_else(|| err(format!("unknown state {name:?}")))?;
        compiled.session.state(slot)
    }
}

impl<'model, F> StatefulSession<'model, F>
where
    F: Fn(&mut StateCx, &[Tensor]) -> Result<Vec<Tensor>>,
{
    /// Begin one state transaction. The returned step exclusively borrows this
    /// session until it is evaluated or dropped.
    ///
    /// ```compile_fail
    /// # use rxla_core::{Result, StateCx, StatefulModel, Tensor};
    /// # fn demo(x: Tensor) -> Result<()> {
    /// let model = StatefulModel::new(|_: &mut StateCx, xs: &[Tensor]| Ok(vec![xs[0].clone()]));
    /// let mut session = model.session();
    /// let first = session.call(std::slice::from_ref(&x))?;
    /// let second = session.call(std::slice::from_ref(&x))?;
    /// drop((first, second));
    /// # Ok(()) }
    /// ```
    pub fn call<'session>(
        &'session mut self,
        inputs: &[Tensor],
    ) -> Result<StateStep<'session, 'model, F>> {
        let signature = inputs.iter().map(Tensor::ty).collect::<Vec<_>>();
        let (pending, bindings, input_fingerprint) = match &self.compiled {
            Some(compiled) => {
                let trace = InputTrace::capture(inputs)?;
                let input_fingerprint = trace.fingerprint.clone();
                if compiled.signature != signature
                    || compiled.input_fingerprint != input_fingerprint
                {
                    return Err(err("stateful session input program specialization changed"));
                }
                (None, trace.bindings, input_fingerprint)
            }
            None => {
                let (mut cx, symbolic_inputs, bindings, fingerprint) = StateCx::new(inputs)?;
                let outputs = (self.model.build)(&mut cx, &symbolic_inputs)?;
                (Some(cx.finish(&outputs)?), bindings, fingerprint)
            }
        };
        Ok(StateStep {
            owner: self,
            inputs: bindings,
            signature,
            input_fingerprint,
            pending,
        })
    }
}

/// A single-use state transaction. Dropping it performs no execution or commit.
pub struct StateStep<'session, 'model, F> {
    owner: &'session mut StatefulSession<'model, F>,
    inputs: Vec<Tensor>,
    signature: Vec<TensorType>,
    input_fingerprint: String,
    pending: Option<PendingSpecialization>,
}

impl<F> StateStep<'_, '_, F> {
    pub fn eval(mut self, runtime: &mut Runtime) -> Result<Vec<Tensor>> {
        let input_buffers = self
            .inputs
            .iter()
            .map(|input| input.to_buffer(runtime.client()))
            .collect::<Result<Vec<_>>>()?;

        if let Some(pending) = self.pending.take() {
            let named_slots = pending.slots.clone();
            let program = runtime.compile_state_graph(&pending.graph, &pending.outputs)?;
            let slots = named_slots
                .iter()
                .map(|(_, slot)| slot.clone())
                .collect::<Vec<_>>();
            let mut initial = program.zero_state(&slots)?;
            let mut overrides = std::mem::take(&mut self.owner.initial);
            for (name, buffer) in overrides.iter() {
                if !named_slots.iter().any(|(declared, _)| declared == name) {
                    return Err(err(format!(
                        "initial state {name:?} is not declared by the model"
                    )));
                }
                if !buffer.belongs_to(program.client()) {
                    return Err(err(format!(
                        "initial state {name:?} belongs to another client"
                    )));
                }
            }
            for (position, (name, _)) in named_slots.iter().enumerate() {
                if let Some(buffer) = overrides.remove(name) {
                    initial[position].1 = buffer;
                }
            }
            let session = program.session(initial)?;
            self.owner.compiled = Some(CompiledState {
                signature: self.signature.clone(),
                input_fingerprint: self.input_fingerprint.clone(),
                output_count: pending.outputs.len(),
                session,
                slots: named_slots,
            });
        }

        let compiled = self.owner.compiled.as_mut().expect("initialized above");
        let values = compiled.session.run(
            &input_buffers
                .iter()
                .map(std::rc::Rc::as_ref)
                .collect::<Vec<_>>(),
        )?;
        if values.len() != compiled.output_count {
            return Err(err("stateful program returned an unexpected output count"));
        }
        values.into_iter().map(Tensor::materialized).collect()
    }

    pub fn eval_one(self, runtime: &mut Runtime) -> Result<Tensor> {
        let mut values = self.eval(runtime)?;
        if values.len() != 1 {
            return Err(err("stateful step does not have exactly one output"));
        }
        Ok(values.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_effects_discharge_to_hidden_results() {
        let input = Tensor::from_slice([], DType::F32, [2.0]).unwrap();
        let (mut cx, inputs, _, _) = StateCx::new(&[input]).unwrap();
        let state = cx.state("count", &[], DType::I32).unwrap();
        let same = cx.state("count", &[], DType::I32).unwrap();
        assert_eq!(state.slot.identity(), same.slot.identity());
        assert!(cx.state("count", &[1], DType::I32).is_err());

        let count = state.read(&cx).unwrap();
        state
            .write(&mut cx, &count.wrapping_add_scalar(1).unwrap())
            .unwrap();
        let pending = cx.finish(&[inputs[0].clone()]).unwrap();
        let prepared = pending.graph.prepare(&pending.outputs).unwrap();

        assert_eq!(prepared.output_spec(0).unwrap().dtype, DType::F32);
        assert_eq!(
            prepared.state_type(&pending.slots[0].1).unwrap(),
            (DType::I32, vec![])
        );
        assert_eq!(prepared.input_indices(), [0]);
    }

    #[test]
    fn lazy_input_dataflow_is_inlined_and_inplace_state_advances_ssa() {
        let left = Tensor::from_slice([], DType::F32, [2.0]).unwrap();
        let right = Tensor::from_slice([], DType::F32, [3.0]).unwrap();
        let fused_input = left.add(&right).unwrap();
        let (mut cx, inputs, bindings, _) = StateCx::new(&[fused_input]).unwrap();
        let accumulator = cx.state("accumulator", &[], DType::F32).unwrap();
        let before = accumulator.read(&cx).unwrap();
        accumulator.add_(&mut cx, &inputs[0]).unwrap();
        let after = accumulator.read(&cx).unwrap();
        assert_ne!(before.node_id(), after.node_id());

        let pending = cx.finish(&[after]).unwrap();
        let prepared = pending.graph.prepare(&pending.outputs).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(prepared.input_indices(), [0, 1]);
    }

    #[test]
    fn scopes_make_state_paths_stable_and_distinct() {
        let (mut cx, _, _, _) = StateCx::new(&[]).unwrap();
        let left = cx
            .scope("left", |cx| cx.state("cache", &[2], DType::F32))
            .unwrap();
        let right = cx
            .scope("right", |cx| cx.state("cache", &[2], DType::F32))
            .unwrap();
        assert_eq!(left.path(), "left.cache");
        assert_eq!(right.path(), "right.cache");
        assert_ne!(left.slot.identity(), right.slot.identity());
    }

    #[test]
    fn state_only_models_discharge_without_visible_results() {
        let (mut cx, _, _, _) = StateCx::new(&[]).unwrap();
        let counter = cx.state("counter", &[], DType::I32).unwrap();
        let next = counter.read(&cx).unwrap().wrapping_add_scalar(1).unwrap();
        counter.write(&mut cx, &next).unwrap();
        let pending = cx.finish(&[]).unwrap();
        let prepared = pending.graph.prepare(&pending.outputs).unwrap();

        assert!(prepared.output_spec(0).is_none());
        assert_eq!(
            prepared.state_type(&pending.slots[0].1).unwrap(),
            (DType::I32, vec![])
        );
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn evaluated_steps_commit_state_once() {
        let model = StatefulModel::new(|cx: &mut StateCx, inputs: &[Tensor]| {
            let count = cx.state("count", &[], DType::I32)?;
            let old = count.read(cx)?;
            count.write(cx, &old.wrapping_add_scalar(1)?)?;
            Ok(vec![inputs[0].add(&old.to_f32()?)?])
        });
        let mut session = model.session();
        let mut runtime =
            unsafe { Runtime::load(std::env::var("PJRT_PLUGIN_PATH").expect("PJRT_PLUGIN_PATH")) }
                .unwrap();
        let input = Tensor::from_slice([], DType::F32, [2.0])
            .unwrap()
            .add(&Tensor::from_slice([], DType::F32, [3.0]).unwrap())
            .unwrap();

        let first = session
            .call(std::slice::from_ref(&input))
            .unwrap()
            .eval_one(&mut runtime)
            .unwrap();
        assert_eq!(first.to_vec::<f32>().unwrap(), [5.0]);
        assert_eq!(
            session.state("count").unwrap().to_vec::<i32>().unwrap(),
            [1]
        );

        let next_input = Tensor::from_slice([], DType::F32, [5.0])
            .unwrap()
            .add(&Tensor::from_slice([], DType::F32, [7.0]).unwrap())
            .unwrap();
        let second = session
            .call(std::slice::from_ref(&next_input))
            .unwrap()
            .eval_one(&mut runtime)
            .unwrap();
        assert_eq!(second.to_vec::<f32>().unwrap(), [13.0]);
        assert_eq!(
            session.state("count").unwrap().to_vec::<i32>().unwrap(),
            [2]
        );
        assert_eq!(runtime.stats().misses, 1);
    }
}
