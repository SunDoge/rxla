use rxla_core::state_tree::{StateTree, states};
use rxla_core::{
    CacheLimits, Client, Compiler, F32, I32, State, StateGraph, StateSlot, impl_state_tree,
};

struct Counter {
    value: State<I32>,
}
impl_state_tree!(Counter { state value => "value" });
struct Model {
    mean: State<F32>,
    optional: Option<State<I32>>,
    primary: Counter,
    extra: Option<Counter>,
    replicas: Vec<Counter>,
}
impl_state_tree!(Model {
    state mean => "mean",
    optional_state optional => "optional",
    tree primary => "primary",
    optional_tree extra => "extra",
    trees replicas => "replicas",
});
struct Empty;
impl_state_tree!(Empty {});

#[test]
fn pointer_wrappers_preserve_dynamic_tree_names_and_aliases() {
    fn collect<T: StateTree>(tree: T) -> Vec<String> {
        states(&tree).unwrap().into_iter().map(|e| e.name).collect()
    }
    let mut g = StateGraph::default();
    let value = State::<I32>::new(&mut g, &[]).unwrap();
    let mut counter = Counter {
        value: value.clone(),
    };
    assert_eq!(collect(&counter), ["value"]);
    assert_eq!(collect(&mut counter), ["value"]);
    let boxed: Box<dyn StateTree> = Box::new(Counter {
        value: value.clone(),
    });
    assert_eq!(collect(&boxed), ["value"]);
    assert_eq!(states(&boxed).unwrap()[0].name, "value");
    let shared: std::rc::Rc<dyn StateTree> = std::rc::Rc::new(Counter {
        value: value.clone(),
    });
    assert_eq!(collect(shared.clone()), ["value"]);
    // Arc is only a pointer forwarding check; this does not make the tree Send.
    #[allow(clippy::arc_with_non_send_sync)]
    let atomic: std::sync::Arc<dyn StateTree> = std::sync::Arc::new(Counter { value });
    assert_eq!(collect(atomic.clone()), ["value"]);
    struct Dynamic {
        primary: Box<dyn StateTree>,
        optional: Option<std::rc::Rc<dyn StateTree>>,
        layers: Vec<Box<dyn StateTree>>,
    }
    impl_state_tree!(Dynamic {
        tree primary => "primary",
        optional_tree optional => "optional",
        trees layers => "layers",
    });
    let mut dynamic = Dynamic {
        primary: boxed,
        optional: Some(shared),
        layers: vec![Box::new(Empty), Box::new(atomic)],
    };
    let entries = states(&dynamic).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "primary.value");
    assert_eq!(entries[0].aliases, ["optional.value", "layers.1.value"]);
    dynamic.optional = None;
    assert_eq!(states(&dynamic).unwrap()[0].aliases, ["layers.1.value"]);
    let bad: Box<dyn StateTree> =
        Box::new(Manual(vec![("bad..path".into(), g.state(&[]).unwrap())]));
    assert!(states(&bad).is_err());
}

fn model(g: &mut StateGraph) -> Model {
    let counter = State::<I32>::new(g, &[]).unwrap();
    Model {
        mean: State::<F32>::new(g, &[2]).unwrap(),
        optional: Some(counter.clone()),
        primary: Counter {
            value: counter.clone(),
        },
        extra: Some(Counter {
            value: counter.clone(),
        }),
        replicas: vec![
            Counter { value: counter },
            Counter {
                value: State::<I32>::new(g, &[]).unwrap(),
            },
        ],
    }
}

#[test]
fn stable_names_aliases_and_optional_members() {
    let mut g = StateGraph::default();
    let mut m = model(&mut g);
    let entries = states(&m).unwrap();
    assert_eq!(
        entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        ["mean", "optional", "replicas.1.value"]
    );
    assert_eq!(
        entries[1].aliases,
        ["primary.value", "extra.value", "replicas.0.value"]
    );
    m.optional = None;
    m.extra = None;
    let entries = states(&m).unwrap();
    assert_eq!(entries[1].name, "primary.value");
    assert_eq!(entries[1].aliases, ["replicas.0.value"]);
    assert!(states(&Empty).unwrap().is_empty());
}

struct Manual(Vec<(String, StateSlot)>);
impl StateTree for Manual {
    fn visit_states(&self, visitor: &mut dyn FnMut(&str, &StateSlot)) {
        for (name, slot) in &self.0 {
            visitor(name, slot);
        }
    }
}
#[test]
fn rejects_bad_paths_but_distinguishes_foreign_identities() {
    let mut g = StateGraph::default();
    let slot = g.state(&[]).unwrap();
    for name in ["", ".a", "a.", "a..b"] {
        assert!(states(&Manual(vec![(name.into(), slot.clone())])).is_err());
    }
    assert!(
        states(&Manual(vec![
            ("a".into(), slot.clone()),
            ("a".into(), slot.clone())
        ]))
        .is_err()
    );
    let foreign = StateGraph::default().state(&[]).unwrap();
    let entries = states(&Manual(vec![("a".into(), slot), ("b".into(), foreign)])).unwrap();
    assert_eq!(entries.len(), 2); // program membership is checked later
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_tree_initializes_complete_unique_state_and_commits() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut g = StateGraph::default();
    let m = Box::new(model(&mut g));
    let entries = states(&m).unwrap();
    let slots: Vec<_> = entries.iter().map(|e| e.slot.clone()).collect();
    let mut tx = g.transaction();
    let next_mean = tx.read(&m.mean).unwrap().add_scalar(2.).unwrap();
    let next_counter = tx
        .read(&m.primary.value)
        .unwrap()
        .wrapping_add_scalar(1)
        .unwrap();
    tx.set(&m.mean, &next_mean)
        .unwrap()
        .set(&m.primary.value, &next_counter)
        .unwrap();
    tx.commit().unwrap();
    let program = g
        .compile(&mut compiler, &[m.mean.read(&g).unwrap()])
        .unwrap();
    assert!(program.zero_state(&slots[..2]).is_err());
    let mut foreign = slots.clone();
    foreign[2] = StateGraph::default().state_i32(&[]).unwrap();
    assert!(program.zero_state(&foreign).is_err());
    let mut session = program
        .session(program.zero_state(&slots).unwrap())
        .unwrap();
    for step in 1..=3 {
        assert_eq!(
            session.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(),
            [2. * step as f32; 2]
        );
        assert_eq!(
            session
                .state(m.primary.value.as_slot())
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [step]
        );
        assert_eq!(
            session
                .state(m.replicas[1].value.as_slot())
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [0]
        );
    }
    assert_eq!(compiler.stats().misses, 1);
}
