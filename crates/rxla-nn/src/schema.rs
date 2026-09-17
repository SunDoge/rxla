//! Immutable model effect schema and its typed parameter identities.

use crate::{Initializer, MissingInitializerSnafu, ParameterId, ParameterSelection, Result};
use rxla_core::{Buffer, Client, DType};
use snafu::OptionExt;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_SCHEMA_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_schema_identity() -> u64 {
    let identity = NEXT_SCHEMA_IDENTITY.fetch_add(1, Ordering::Relaxed);
    assert_ne!(identity, 0, "model schema identity space exhausted");
    identity
}

/// One immutable parameter declaration in a traced model schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpec {
    pub(crate) path: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) storage_dtype: DType,
    pub(crate) compute_dtype: DType,
    pub(crate) initializer: Option<Initializer>,
}

impl ParameterSpec {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.storage_dtype
    }

    /// Element type required from a checkpoint or parameter initializer.
    pub fn storage_dtype(&self) -> DType {
        self.storage_dtype
    }

    /// Element type observed by tensor operations after an explicit conversion.
    pub fn compute_dtype(&self) -> DType {
        self.compute_dtype
    }

    pub fn initializer(&self) -> Option<Initializer> {
        self.initializer
    }
}

/// One immutable input declaration in a traced model schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInputSpec {
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
}

/// One named resident state declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateSpec {
    pub(crate) path: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
    pub(crate) initializer: Initializer,
}

impl StateSpec {
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn initializer(&self) -> Initializer {
        self.initializer
    }
}

impl ModelInputSpec {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// One runtime argument in the model ABI, in effect declaration order.
///
/// The indices refer to [`ModelSchema::inputs`] and
/// [`ModelSchema::parameters`], respectively. This is the authoritative
/// binding order before unreachable arguments are compacted at lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelArgument {
    Input(usize),
    Parameter(usize),
    State(usize),
}

/// Ordered, immutable declarations produced by [`crate::init`].
#[derive(Debug)]
struct ModelSchemaData {
    identity: u64,
    pub(crate) inputs: Vec<ModelInputSpec>,
    pub(crate) parameters: Vec<ParameterSpec>,
    pub(crate) indices: BTreeMap<String, usize>,
    pub(crate) arguments: Vec<ModelArgument>,
    pub(crate) states: Vec<StateSpec>,
}

impl Default for ModelSchemaData {
    fn default() -> Self {
        Self {
            identity: next_schema_identity(),
            inputs: Vec::new(),
            parameters: Vec::new(),
            indices: BTreeMap::new(),
            arguments: Vec::new(),
            states: Vec::new(),
        }
    }
}

impl Clone for ModelSchemaData {
    fn clone(&self) -> Self {
        Self {
            identity: next_schema_identity(),
            inputs: self.inputs.clone(),
            parameters: self.parameters.clone(),
            indices: self.indices.clone(),
            arguments: self.arguments.clone(),
            states: self.states.clone(),
        }
    }
}

impl PartialEq for ModelSchemaData {
    fn eq(&self, other: &Self) -> bool {
        self.inputs == other.inputs
            && self.parameters == other.parameters
            && self.indices == other.indices
            && self.arguments == other.arguments
            && self.states == other.states
    }
}

impl Eq for ModelSchemaData {}

/// Cheaply cloned identity-bearing handle to one immutable effect schema.
///
/// Schema discovery mutates a uniquely owned handle. Once returned, clones
/// share the same allocation so selections can retain provenance without a
/// borrow or a deep copy.
#[derive(Clone, Debug, Default)]
pub struct ModelSchema(Arc<ModelSchemaData>);

impl PartialEq for ModelSchema {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ModelSchema {}

impl ModelSchema {
    /// Materialize every parameter from its declaration-time initializer.
    ///
    /// Seeds are derived from `seed` and the stable parameter path, so adding
    /// an unrelated parameter does not perturb existing initial values.
    pub fn initialize(&self, client: &Client, seed: u64) -> Result<Vec<(String, Buffer)>> {
        let declarations = self
            .parameters()
            .iter()
            .map(|parameter| {
                let initializer = parameter.initializer.context(MissingInitializerSnafu {
                    path: parameter.path(),
                })?;
                initializer.validate(parameter.path(), parameter.shape(), parameter.dtype())?;
                Ok((parameter, initializer))
            })
            .collect::<Result<Vec<_>>>()?;
        declarations
            .into_iter()
            .map(|(parameter, initializer)| {
                Ok((
                    parameter.path.clone(),
                    initializer.initialize(
                        client,
                        parameter.path(),
                        parameter.shape(),
                        parameter.dtype(),
                        stable_seed(seed, parameter.path()),
                    )?,
                ))
            })
            .collect()
    }

    /// Input ABI in the order inputs are requested by the model trace.
    pub fn inputs(&self) -> &[ModelInputSpec] {
        &self.0.inputs
    }

    pub fn parameters(&self) -> &[ParameterSpec] {
        &self.0.parameters
    }

    pub fn states(&self) -> &[StateSpec] {
        &self.0.states
    }

    pub fn get(&self, path: &str) -> Option<&ParameterSpec> {
        self.0
            .indices
            .get(path)
            .and_then(|&index| self.0.parameters.get(index))
    }

    /// Stable identity of a parameter within this immutable schema.
    pub fn parameter_id(&self, path: &str) -> Option<ParameterId> {
        self.0
            .indices
            .get(path)
            .copied()
            .map(|index| ParameterId::new(self.0.identity, index))
    }

    /// Parameter declaration identified by [`ParameterId`].
    pub fn parameter(&self, id: ParameterId) -> Option<&ParameterSpec> {
        (id.schema_identity() == self.0.identity)
            .then(|| self.0.parameters.get(id.index()))
            .flatten()
    }

    /// Start an ordered selection containing every model parameter.
    ///
    /// Trainability is deliberately selected from a schema at training time;
    /// it is not a permanent `requires_grad` bit on the model definition.
    pub fn select_all(&self) -> ParameterSelection {
        ParameterSelection::all(self)
    }

    /// Select parameters below one lexical scope (`scope` itself included).
    pub fn select_under(&self, scope: &str) -> ParameterSelection {
        self.select_all().under(scope)
    }

    /// Full runtime ABI in source effect declaration order.
    pub fn arguments(&self) -> &[ModelArgument] {
        &self.0.arguments
    }

    pub(crate) fn same_identity(&self, other: &Self) -> bool {
        self.0.identity == other.0.identity
    }

    pub(crate) fn id_at(&self, index: usize) -> ParameterId {
        ParameterId::new(self.0.identity, index)
    }

    pub(crate) fn push_parameter(&mut self, spec: ParameterSpec) -> usize {
        let data = Arc::make_mut(&mut self.0);
        let index = data.parameters.len();
        data.indices.insert(spec.path.clone(), index);
        data.parameters.push(spec);
        data.arguments.push(ModelArgument::Parameter(index));
        index
    }

    pub(crate) fn push_input(&mut self, spec: ModelInputSpec) -> usize {
        let data = Arc::make_mut(&mut self.0);
        let index = data.inputs.len();
        data.inputs.push(spec);
        data.arguments.push(ModelArgument::Input(index));
        index
    }

    pub(crate) fn push_state(&mut self, spec: StateSpec) -> usize {
        let data = Arc::make_mut(&mut self.0);
        let index = data.states.len();
        data.states.push(spec);
        data.arguments.push(ModelArgument::State(index));
        index
    }
}

pub(crate) fn stable_seed(mut seed: u64, path: &str) -> u64 {
    for byte in path.bytes() {
        seed ^= u64::from(byte);
        seed = seed.wrapping_mul(0x100_0000_01b3);
    }
    seed
}
