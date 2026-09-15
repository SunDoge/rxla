//! Immutable model effect schema and its typed parameter identities.

use crate::{ParameterId, ParameterSelection};
use rxla_core::DType;
use std::collections::BTreeMap;

/// One immutable parameter declaration in a traced model schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpec {
    pub(crate) path: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
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
}

/// One immutable input declaration in a traced model schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInputSpec {
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
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
}

/// Ordered, immutable declarations produced by [`crate::init`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParamSchema {
    pub(crate) inputs: Vec<ModelInputSpec>,
    pub(crate) parameters: Vec<ParameterSpec>,
    pub(crate) indices: BTreeMap<String, usize>,
    pub(crate) arguments: Vec<ModelArgument>,
}

impl ParamSchema {
    /// Input ABI in the order inputs are requested by the model trace.
    pub fn inputs(&self) -> &[ModelInputSpec] {
        &self.inputs
    }

    pub fn parameters(&self) -> &[ParameterSpec] {
        &self.parameters
    }

    pub fn get(&self, path: &str) -> Option<&ParameterSpec> {
        self.indices
            .get(path)
            .and_then(|&index| self.parameters.get(index))
    }

    /// Stable identity of a parameter within this immutable schema.
    pub fn parameter_id(&self, path: &str) -> Option<ParameterId> {
        self.indices.get(path).copied().map(ParameterId::from_index)
    }

    /// Parameter declaration identified by [`ParameterId`].
    pub fn parameter(&self, id: ParameterId) -> Option<&ParameterSpec> {
        self.parameters.get(id.index())
    }

    /// Start an ordered selection containing every model parameter.
    ///
    /// Trainability is deliberately selected from a schema at training time;
    /// it is not a permanent `requires_grad` bit on the model definition.
    pub fn select_all(&self) -> ParameterSelection<'_> {
        ParameterSelection::all(self)
    }

    /// Select parameters below one lexical scope (`scope` itself included).
    pub fn select_under(&self, scope: &str) -> ParameterSelection<'_> {
        self.select_all().under(scope)
    }

    /// Full runtime ABI in source effect declaration order.
    pub fn arguments(&self) -> &[ModelArgument] {
        &self.arguments
    }
}
