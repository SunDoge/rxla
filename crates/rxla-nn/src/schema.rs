//! Immutable model effect schema and its typed parameter identities.

use crate::{Initializer, MissingInitializerSnafu, ParameterId, ParameterSelection, Result};
use rxla_core::{Buffer, Client, DType};
use snafu::OptionExt;
use std::{collections::BTreeMap, sync::Arc};

/// One immutable parameter declaration in a traced model schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpec {
    pub(crate) path: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
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
        self.dtype
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
/// The indices refer to [`ParamSchema::inputs`] and
/// [`ParamSchema::parameters`], respectively. This is the authoritative
/// binding order before unreachable arguments are compacted at lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelArgument {
    Input(usize),
    Parameter(usize),
    State(usize),
}

/// Ordered, immutable declarations produced by [`crate::init`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ParamSchemaData {
    pub(crate) inputs: Vec<ModelInputSpec>,
    pub(crate) parameters: Vec<ParameterSpec>,
    pub(crate) indices: BTreeMap<String, usize>,
    pub(crate) arguments: Vec<ModelArgument>,
    pub(crate) states: Vec<StateSpec>,
}

/// Cheaply cloned identity-bearing handle to one immutable effect schema.
///
/// Schema discovery mutates a uniquely owned handle. Once returned, clones
/// share the same allocation so selections can retain provenance without a
/// borrow or a deep copy.
#[derive(Clone, Debug, Default)]
pub struct ParamSchema(Arc<ParamSchemaData>);

impl PartialEq for ParamSchema {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ParamSchema {}

impl ParamSchema {
    /// Materialize every parameter from its declaration-time initializer.
    ///
    /// Seeds are derived from `seed` and the stable parameter path, so adding
    /// an unrelated parameter does not perturb existing initial values.
    pub fn initialize(&self, client: &Client, seed: u64) -> Result<Vec<(String, Buffer)>> {
        self.parameters()
            .iter()
            .map(|parameter| {
                let initializer = parameter.initializer.context(MissingInitializerSnafu {
                    path: parameter.path(),
                })?;
                let parameter_seed = stable_seed(seed, parameter.path());
                Ok((
                    parameter.path.clone(),
                    initializer.initialize(
                        client,
                        parameter.path(),
                        parameter.shape(),
                        parameter.dtype(),
                        parameter_seed,
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
            .map(ParameterId::from_index)
    }

    /// Parameter declaration identified by [`ParameterId`].
    pub fn parameter(&self, id: ParameterId) -> Option<&ParameterSpec> {
        self.0.parameters.get(id.index())
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
        Arc::ptr_eq(&self.0, &other.0)
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

fn stable_seed(mut seed: u64, path: &str) -> u64 {
    for byte in path.bytes() {
        seed ^= u64::from(byte);
        seed = seed.wrapping_mul(0x100_0000_01b3);
    }
    seed
}
