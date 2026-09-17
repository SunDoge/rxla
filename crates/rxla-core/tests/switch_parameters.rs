use rxla_core::{CacheLimits, Client, Compiler, StateGraph};
use std::sync::Arc;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_parameter_switch_is_atomic_and_dtype_checked() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut a = StateGraph::default();
    let p = a.parameter(&[]).unwrap();
    let s = a.state(&[]).unwrap();
    let y = a.read(&s).unwrap().add(p.tensor()).unwrap();
    a.write(&s, &y).unwrap();
    let first = a.compile(&mut compiler, &[y]).unwrap();
    let mut b = StateGraph::default();
    b.input(&[3]).unwrap(); // Pruned input before the destination weight.
    let q = b.parameter_bf16_as_f32(&[]).unwrap();
    let t = b.state(&[]).unwrap();
    let y = b.read(&t).unwrap().add(q.tensor()).unwrap();
    b.write(&t, &y).unwrap();
    let second = b.compile_pruned(&mut compiler, &[y]).unwrap();
    let mut session = first
        .session(vec![(s.clone(), client.buffer(&[], &[0.]).unwrap())])
        .unwrap();
    let old = Arc::new(client.buffer(&[], &[2.]).unwrap());
    let new = Arc::new(
        client
            .buffer(&[], &[rxla_core::bf16::from_bits(0x4040)])
            .unwrap(),
    );
    session
        .bind_parameters(vec![(p.clone(), old.clone())])
        .unwrap();
    let mapping = [(s.clone(), t.clone())];
    for invalid in [
        vec![(p, old.clone())],
        vec![(q.clone(), old)],
        vec![(q.clone(), new.clone()), (q.clone(), new.clone())],
    ] {
        assert!(
            session
                .switch_program_parameters(&second, &mapping, invalid)
                .is_err()
        );
        assert_eq!(session.state(&s).unwrap().to_vec::<f32>().unwrap(), [0.]);
    }
    assert_eq!(session.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [2.]);
    session
        .switch_program_parameters(&second, &mapping, vec![(q, new)])
        .unwrap();
    assert_eq!(session.run(&[]).unwrap()[0].to_vec::<f32>().unwrap(), [5.]);
    assert!(session.state(&s).is_err());
    assert_eq!(compiler.stats().misses, 2);
}
