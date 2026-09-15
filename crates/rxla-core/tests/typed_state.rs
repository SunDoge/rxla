use rxla_core::{CacheLimits, Client, Compiler, F32, I32, State, StateGraph, StateUpdates};

#[test]
fn typed_state_rejects_wrong_erased_types_shapes_and_owners() {
    let mut g = StateGraph::default();
    let f = State::<F32>::new(&mut g, &[2]).unwrap();
    let i = State::<I32>::new(&mut g, &[]).unwrap();
    assert!(State::<I32>::from_slot(&g, f.as_slot().clone()).is_err());
    assert!(State::<F32>::from_slot(&g, i.as_slot().clone()).is_err());
    let other = StateGraph::default();
    assert!(f.read(&other).is_err());
    assert!(State::<I32>::from_slot(&other, i.as_slot().clone()).is_err());
    let value = g.constant(&[], &[3.]).unwrap();
    assert!(f.write(&mut g, &value).is_err());
    let foreign = other.constant(&[2], &[1., 2.]).unwrap();
    assert!(f.write(&mut g, &foreign).is_err());
    let valid = g.constant(&[2], &[1., 2.]).unwrap();
    let alias = State::<F32>::from_slot(&g, f.clone().into_slot()).unwrap();
    alias.write(&mut g, &valid).unwrap();
    assert_eq!(f.read(&g).unwrap().shape(), [2]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_typed_state_guarded_versions_and_session_bridge() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for mode in 0..5 {
        let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
        let mut g = StateGraph::default();
        let accept = g.input(&[]).unwrap();
        let f = State::<F32>::new(&mut g, &[2]).unwrap();
        let i = State::<I32>::new(&mut g, &[]).unwrap();
        let next_f = f.read(&g).unwrap().add_scalar(2.).unwrap();
        let next_i = i.read(&g).unwrap().wrapping_add_scalar(1).unwrap();
        if mode == 4 {
            g.transaction()
                .with(&f, &next_f)
                .unwrap()
                .with(&i, &next_i)
                .unwrap()
                .commit_if(&accept)
                .unwrap();
        } else if mode == 3 {
            let mut tx = g.transaction();
            let next_f = tx.read(&f).unwrap().add_scalar(2.).unwrap();
            let next_i = tx.read(&i).unwrap().wrapping_add_scalar(1).unwrap();
            tx.set(&f, &next_f).unwrap().set(&i, &next_i).unwrap();
            tx.commit_if(&accept).unwrap();
        } else if mode == 2 {
            StateUpdates::new()
                .with(&f, &next_f)
                .with(&i, &next_i)
                .commit_if(&mut g, &accept)
                .unwrap();
        } else if mode == 1 {
            let mut updates = StateUpdates::new();
            updates.set(&f, &next_f).set(&i, &next_i);
            updates.commit_if(&mut g, &accept).unwrap();
        } else {
            f.write_if(&mut g, &next_f, &accept).unwrap();
            i.write_if(&mut g, &next_i, &accept).unwrap();
        }
        // The erased bridge continues to work with the existing program/session API.
        let program = g.compile(&mut compiler, &[f.read(&g).unwrap()]).unwrap();
        let mut session = program
            .session(vec![
                (f.as_slot().clone(), client.buffer(&[2], &[1., 3.]).unwrap()),
                (
                    i.as_slot().clone(),
                    client.buffer(&[], &[i32::MAX]).unwrap(),
                ),
            ])
            .unwrap();
        for (condition, expected_f, expected_i) in [
            (0., [1., 3.], i32::MAX),
            (1., [3., 5.], i32::MIN),
            (-0., [3., 5.], i32::MIN),
            (-1., [5., 7.], i32::MIN + 1),
        ] {
            let mask = client.buffer(&[], &[condition]).unwrap();
            let out = session.run(&[&mask]).unwrap();
            assert_eq!(out[0].to_vec::<f32>().unwrap(), expected_f);
            assert_eq!(
                session.state(f.as_slot()).unwrap().to_vec::<f32>().unwrap(),
                expected_f
            );
            assert_eq!(
                session.state(i.as_slot()).unwrap().to_vec::<i32>().unwrap(),
                [expected_i]
            );
            session = program.session(session.into_state()).unwrap();
        }
        assert_eq!(compiler.stats().misses, 1);
    }
}
