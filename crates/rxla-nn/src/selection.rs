//! Ordered parameter selection for partial differentiation and updates.

use super::{ModelSchema, ParameterSpec};

/// Stable parameter identity within one immutable [`ModelSchema`].
///
/// The identity carries its originating schema, so using an ID with an
/// independently traced schema cannot silently address the same numeric slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId {
    schema: u64,
    index: usize,
}

impl ParameterId {
    pub(crate) const fn new(schema: u64, index: usize) -> Self {
        Self { schema, index }
    }

    pub(crate) const fn index(self) -> usize {
        self.index
    }

    pub(crate) const fn schema_identity(self) -> u64 {
        self.schema
    }
}

/// An ordered, cheaply owned view of parameters from one immutable schema.
///
/// Selections preserve schema order, so gradient and optimizer argument order
/// remains deterministic. Filters compose by intersection; exclusions subtract
/// from the current selection.
#[derive(Clone, Debug)]
pub struct ParameterSelection {
    schema: ModelSchema,
    ids: Vec<ParameterId>,
}

impl ParameterSelection {
    pub(crate) fn all(schema: &ModelSchema) -> Self {
        Self {
            schema: schema.clone(),
            ids: (0..schema.parameters().len())
                .map(|index| schema.id_at(index))
                .collect(),
        }
    }

    /// Keep parameters matching `predicate`, preserving schema order.
    pub fn matching(
        mut self,
        mut predicate: impl FnMut(ParameterId, &ParameterSpec) -> bool,
    ) -> Self {
        self.ids
            .retain(|&id| predicate(id, self.schema.parameter(id).expect("id came from schema")));
        self
    }

    /// Keep parameters declared at or below a dot-delimited lexical scope.
    pub fn under(self, scope: &str) -> Self {
        self.matching(|_, parameter| path_is_under(parameter.path(), scope))
    }

    /// Remove parameters declared at or below a lexical scope.
    pub fn excluding(self, scope: &str) -> Self {
        self.matching(|_, parameter| !path_is_under(parameter.path(), scope))
    }

    pub fn ids(&self) -> &[ParameterId] {
        &self.ids
    }

    pub fn parameters(&self) -> impl ExactSizeIterator<Item = (ParameterId, &ParameterSpec)> + '_ {
        self.ids.iter().copied().map(|id| {
            let parameter = self.schema.parameter(id).expect("id came from schema");
            (id, parameter)
        })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn contains(&self, id: ParameterId) -> bool {
        self.ids.binary_search(&id).is_ok()
    }

    /// The immutable schema that gives every selected ID its meaning.
    pub fn schema(&self) -> &ModelSchema {
        &self.schema
    }
}

pub(crate) fn path_is_under(path: &str, scope: &str) -> bool {
    scope.is_empty()
        || path == scope
        || path
            .strip_prefix(scope)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

#[cfg(test)]
mod tests {
    use crate::{Cx, init};
    use rxla_core::Tensor;

    fn branched(cx: &mut Cx) -> crate::Result<Tensor> {
        let input = cx.input(&[2, 4])?;
        let body = cx.scope("body")?.linear(4).apply(&input)?;
        cx.scope("head")?.linear(2).apply(&body)
    }

    #[test]
    fn scope_selection_is_boundary_aware_and_ordered() {
        let (schema, _) = init(branched).unwrap();
        let body = schema.select_under("body");
        let paths: Vec<_> = body
            .parameters()
            .map(|(_, parameter)| parameter.path())
            .collect();
        assert_eq!(paths, ["body.weight", "body.bias"]);
        assert!(schema.select_under("bo").is_empty());
        assert_eq!(schema.select_all().excluding("body").len(), 2);
    }

    #[test]
    fn ids_round_trip_without_exposing_raw_indices() {
        let (schema, _) = init(branched).unwrap();
        let id = schema.parameter_id("head.weight").unwrap();
        assert_eq!(schema.parameter(id).unwrap().path(), "head.weight");
        let selection = schema.select_under("head");
        assert!(selection.contains(id));
        assert!(selection.schema().parameter(id).is_some());
    }

    #[test]
    fn ids_reject_the_same_numeric_slot_from_another_schema() {
        let (first, _) = init(branched).unwrap();
        let (second, _) = init(branched).unwrap();
        assert_eq!(first, second);

        let first_id = first.parameter_id("body.weight").unwrap();
        let second_id = second.parameter_id("body.weight").unwrap();
        assert_ne!(first_id, second_id);
        assert!(second.parameter(first_id).is_none());
        assert!(!second.select_all().contains(first_id));
    }
}
