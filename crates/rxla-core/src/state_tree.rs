//! Explicit named discovery for symbolic state structures. No implicit updates.
use crate::{Result, StateSlot, err};

/// Object-safe, stable-order state discovery. Implementations must explicitly
/// expose every desired field. Paths use nonempty dot-separated segments.
pub trait StateTree {
    fn visit_states(&self, visitor: &mut dyn FnMut(&str, &StateSlot));
}

// Forward the trait itself, not only method autoderef: wrappers also work with
// generic StateTree bounds and direct coercion to &dyn StateTree. No new paths,
// slot identities, synchronization, or Send/Sync guarantees are introduced.
macro_rules! forward_state_tree {
    ($($wrapper:ty),* $(,)?) => {
        $(impl<T: StateTree + ?Sized> StateTree for $wrapper {
            fn visit_states(&self, visitor: &mut dyn FnMut(&str, &StateSlot)) {
                (**self).visit_states(visitor);
            }
        })*
    };
}
forward_state_tree!(&T, &mut T, Box<T>, std::rc::Rc<T>, std::sync::Arc<T>);

/// One unique slot, with first-visit name and subsequent alias paths. Names are
/// application schema, not durable slot IDs; slot identities are process-local.
pub struct StateEntry {
    pub name: String,
    pub aliases: Vec<String>,
    pub slot: StateSlot,
}

/// Collect named slots, rejecting empty path segments and duplicate names.
/// Shared identities are deduplicated while preserving aliases/traversal order.
/// This does not prove program membership or completeness: use StateProgram's
/// state_layout/zero_state or Session initialization to validate the full schema.
pub fn states(tree: &dyn StateTree) -> Result<Vec<StateEntry>> {
    let mut entries: Vec<StateEntry> = Vec::new();
    let mut identities = std::collections::HashMap::new();
    let mut paths = std::collections::HashSet::new();
    let mut failure = None;
    tree.visit_states(&mut |path, slot| {
        if failure.is_some() {
            return;
        }
        if path.split('.').any(str::is_empty) {
            failure = Some(err("state paths must have nonempty segments"));
            return;
        }
        if !paths.insert(path.to_owned()) {
            failure = Some(err(format!("duplicate state path: {path}")));
            return;
        }
        let next = entries.len();
        let index = *identities.entry(slot.identity()).or_insert(next);
        if index == next {
            entries.push(StateEntry {
                name: path.into(),
                aliases: Vec::new(),
                slot: slot.clone(),
            });
        } else {
            entries[index].aliases.push(path.into());
        }
    });
    match failure {
        Some(error) => Err(error),
        None => Ok(entries),
    }
}

/// Generate explicit state discovery for a concrete type. Supports typed `state`,
/// `optional_state`, nested `tree`, `optional_tree`, and iterable `trees` fields.
/// Fields not listed are not discovered. This does not generate initialization,
/// checkpoint encoding, updates, forward, or parameter discovery. List indices
/// are part of names: reordering changes the schema. Generic impls use a manual
/// StateTree implementation instead. No proc macro/build-time generator is used.
///
/// ```
/// use rxla_core::{State, I32, StateGraph, impl_state_tree};
/// use rxla_core::state_tree::states;
/// struct Counter { steps: State<I32> }
/// impl_state_tree!(Counter { state steps => "steps" });
/// let mut graph = StateGraph::default();
/// let counter = Counter { steps: State::<I32>::new(&mut graph, &[])? };
/// assert_eq!(states(&counter)?[0].name, "steps");
/// # Ok::<(), rxla_core::Error>(())
/// ```
#[macro_export]
macro_rules! impl_state_tree {
    ($model:ty { $($kind:ident $field:ident => $path:literal),* $(,)? }) => {
        impl $crate::state_tree::StateTree for $model {
            fn visit_states(&self, visitor: &mut dyn ::core::ops::FnMut(&str, &$crate::StateSlot)) {
                let _ = &visitor;
                $($crate::impl_state_tree!(@visit self, visitor, $kind, $field, $path);)*
            }
        }
    };
    (@visit $model:ident, $visitor:ident, state, $field:ident, $path:literal) => {
        $visitor($path, $model.$field.as_slot());
    };
    (@visit $model:ident, $visitor:ident, optional_state, $field:ident, $path:literal) => {
        if let ::core::option::Option::Some(state) = &$model.$field { $visitor($path, state.as_slot()); }
    };
    (@visit $model:ident, $visitor:ident, tree, $field:ident, $path:literal) => {{
        use $crate::state_tree::StateTree as _;
        $model.$field.visit_states(&mut |name, slot| $visitor(&::std::format!("{}.{}", $path, name), slot));
    }};
    (@visit $model:ident, $visitor:ident, optional_tree, $field:ident, $path:literal) => {
        if let ::core::option::Option::Some(tree) = &$model.$field {
            use $crate::state_tree::StateTree as _;
            tree.visit_states(&mut |name, slot| $visitor(&::std::format!("{}.{}", $path, name), slot));
        }
    };
    (@visit $model:ident, $visitor:ident, trees, $field:ident, $path:literal) => {
        for (index, tree) in $model.$field.iter().enumerate() {
            use $crate::state_tree::StateTree as _;
            tree.visit_states(&mut |name, slot| $visitor(&::std::format!("{}.{}.{}", $path, index, name), slot));
        }
    };
    (@visit $model:ident, $visitor:ident, $kind:ident, $field:ident, $path:literal) => {
        compile_error!("expected state, optional_state, tree, optional_tree or trees");
    };
}
