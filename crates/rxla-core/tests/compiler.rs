use rxla_core::{CacheLimits, Client, Compiler, Graph, Tensor};
use std::rc::Rc;

fn graph(value: f32) -> (Graph, Tensor) {
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let y = x.add_scalar(value).unwrap();
    (g, y)
}

fn client() -> Client {
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
fn lowered_program_needs_no_plugin_and_rejects_invalid_outputs() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<rxla_core::LoweredProgram>();
    let (g, y) = graph(1.);
    g.prepare(&y).unwrap();
    assert!(g.prepare_outputs(&[]).is_err());
    assert!(Graph::default().prepare(&y).is_err());
    assert!(g.prepare_outputs_pruned(&[]).is_err());
    assert!(Graph::default().prepare_outputs_pruned(&[y]).is_err());
}

#[test]
fn prepared_signature_reports_storage_types_and_compact_parameter_order() {
    use rxla_core::{DType, InputSpec};
    let g = Graph::default();
    let empty = g.input(&[2, 0]).unwrap();
    let index = g.input_i32_scalar().unwrap();
    let bf16 = g.input_bf16_as_f32(&[3]).unwrap();
    let full = g.prepare_outputs(&[empty, index, bf16.clone()]).unwrap();
    assert_eq!(full.input_count(), 3);
    assert_eq!(full.output_count(), 3);
    for (i, (shape, dtype)) in [
        (&[2, 0][..], DType::F32),
        (&[][..], DType::I32),
        (&[3][..], DType::F32),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            full.output_spec(i),
            Some(rxla_core::OutputSpec { shape, dtype })
        );
    }
    assert!(full.output_spec(3).is_none());
    assert!(full.output_spec(usize::MAX).is_none());
    let single = g.prepare(&bf16).unwrap();
    assert_eq!(single.output_spec(0), full.output_spec(2));
    assert!(single.output_spec(1).is_none());
    for (i, (shape, dtype)) in [
        (&[2, 0][..], DType::F32),
        (&[][..], DType::I32),
        (&[3][..], DType::BF16),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(full.input_spec(i), Some(InputSpec { shape, dtype }));
    }
    assert!(full.input_spec(3).is_none());
    assert!(full.input_spec(usize::MAX).is_none());
    let (pruned, mapping) = g.prepare_outputs_pruned(&[bf16.clone(), bf16]).unwrap();
    assert_eq!(mapping, [2]);
    assert_eq!(pruned.input_count(), 1);
    assert_eq!(pruned.output_count(), 2);
    assert_eq!(pruned.input_spec(0), full.input_spec(2));
    let constant = g.constant(&[], &[1.]).unwrap();
    let (constant, _) = g.prepare_outputs_pruned(&[constant]).unwrap();
    assert_eq!(constant.input_count(), 0);
    assert!(constant.input_spec(0).is_none());
    g.input(&[10]).unwrap();
    drop(g);
    assert_eq!(full.input_count(), 3);
    assert_eq!(pruned.input_spec(0).unwrap().dtype, DType::BF16);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_prepared_pruned_inputs_preserve_output_order_and_source_graph() {
    let client = client();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let g = Graph::default();
    g.input(&[100]).unwrap(); // unused prefix
    let x = g.input(&[2]).unwrap();
    g.input_i32(&[]).unwrap(); // unused middle
    let ids = g.input_i32(&[2]).unwrap();
    let surrogate = g.input(&[2]).unwrap(); // backward-only input
    let y = x.with_gradient_of(&surrogate).unwrap();
    let outputs = [ids, y.clone(), y.clone()];
    let before = g.stablehlo_outputs(&outputs).unwrap();
    let (prepared, mapping) = g.prepare_outputs_pruned(&outputs).unwrap();
    assert_eq!(mapping, [1, 3]);
    assert_eq!(g.stablehlo_outputs(&outputs).unwrap(), before);
    let exe = compiler.compile_lowered(&prepared).unwrap();
    assert_eq!(exe.input_count(), 2);
    let input = client.buffer(&[2], &[3., -2.]).unwrap();
    let index = client.buffer(&[2], &[i32::MIN, i32::MAX]).unwrap();
    let actual = exe.execute(&[&input, &index]).unwrap();
    assert_eq!(actual[0].to_vec::<i32>().unwrap(), [i32::MIN, i32::MAX]);
    assert_eq!(actual[1].to_vec::<f32>().unwrap(), [3., -2.]);
    assert_eq!(actual[2].to_vec::<f32>().unwrap(), [3., -2.]);
    // Preparing a forward-only snapshot must not destroy the surrogate AD path.
    let loss = y.mul(&y).unwrap().sum(&[0], false).unwrap();
    let grad = loss.grad(&[surrogate]).unwrap().remove(0);
    let (backward, mapping) = g.prepare_outputs_pruned(&[grad]).unwrap();
    assert_eq!(mapping, [1]);
    assert_eq!(
        compiler
            .compile_lowered(&backward)
            .unwrap()
            .run(&[&[3., -2.]])
            .unwrap(),
        [6., -4.]
    );
    let constant = g.constant(&[], &[7.]).unwrap();
    let (constant, mapping) = g.prepare_outputs_pruned(&[constant]).unwrap();
    assert!(mapping.is_empty());
    assert_eq!(
        compiler
            .compile_lowered(&constant)
            .unwrap()
            .run(&[])
            .unwrap(),
        [7.]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_prepared_snapshot_shares_keys_and_freezes_typed_input_abi() {
    let client = client();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let prepared = {
        let (g, y) = graph(1.);
        let ids = g.input_i32(&[2]).unwrap();
        let outputs = [ids, y];
        let prepared = g.prepare_outputs(&outputs).unwrap();
        let ordinary = compiler.compile_outputs(&g, &outputs).unwrap();
        let before = compiler.stats().compile_time;
        let cached = compiler.compile_lowered(&prepared).unwrap();
        assert!(Rc::ptr_eq(&ordinary, &cached));
        assert_eq!(compiler.stats().compile_time, before);
        assert_eq!(cached.input_count(), 2);
        // Snapshot must not acquire parameters added to the source graph later.
        g.input(&[]).unwrap();
        assert_eq!(
            compiler
                .compile_outputs(&g, &outputs)
                .unwrap()
                .input_count(),
            3
        );
        assert_eq!(
            compiler.compile_lowered(&prepared).unwrap().input_count(),
            2
        );
        prepared
    };
    // All source graph handles have gone; two independent caches can use it.
    let x = client.buffer(&[2], &[3., -4.]).unwrap();
    let ids = client.buffer(&[2], &[i32::MIN, i32::MAX]).unwrap();
    let mut other = Compiler::new(client, CacheLimits::default());
    for compiler in [&mut compiler, &mut other] {
        let exe = compiler.compile_lowered(&prepared).unwrap();
        assert_eq!(exe.input_count(), prepared.input_count());
        assert_eq!(exe.output_count(), prepared.output_count());
        for i in 0..exe.input_count() {
            assert_eq!(exe.input_spec(i), prepared.input_spec(i));
        }
        assert!(exe.execute(&[&x]).is_err());
        assert!(exe.execute(&[&ids, &x]).is_err());
        let output = exe.execute(&[&x, &ids]).unwrap();
        for (i, buffer) in output.iter().enumerate() {
            let spec = prepared.output_spec(i).unwrap();
            assert_eq!(buffer.dimensions().unwrap(), spec.shape);
            assert_eq!(buffer.dtype().unwrap(), spec.dtype);
        }
        assert_eq!(output[0].to_vec::<i32>().unwrap(), [i32::MIN, i32::MAX]);
        assert_eq!(output[1].to_vec::<f32>().unwrap(), [4., -3.]);
    }
    assert_eq!(other.stats().misses, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cache_hits_lru_and_lifetimes() {
    let mut compiler = Compiler::new(
        client(),
        CacheLimits {
            max_entries: 2,
            max_key_bytes: 1 << 20,
        },
    );
    let (a, ya) = graph(1.);
    let (b, yb) = graph(2.);
    let (c, yc) = graph(3.);
    let ea = compiler.compile(&a, &ya).unwrap();
    let eb = compiler.compile(&b, &yb).unwrap();
    let compiled_time = compiler.stats().compile_time;
    assert!(!compiled_time.is_zero());
    // Separately constructed but byte-identical graphs reuse the same executable.
    let (identical, y_identical) = graph(1.);
    assert!(Rc::ptr_eq(
        &ea,
        &compiler.compile(&identical, &y_identical).unwrap()
    ));
    assert_eq!(compiler.stats().compile_time, compiled_time);
    assert_eq!(ea.run(&[&[1., 2.]]).unwrap(), [2., 3.]);
    assert_eq!(ea.run(&[&[10., 20.]]).unwrap(), [11., 21.]);
    let ec = compiler.compile(&c, &yc).unwrap(); // evicts B, not recently touched A
    assert!(Rc::ptr_eq(&ea, &compiler.compile(&a, &ya).unwrap()));
    let eb2 = compiler.compile(&b, &yb).unwrap();
    assert!(!Rc::ptr_eq(&eb, &eb2));
    let stats = compiler.stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.evictions, stats.entries),
        (2, 4, 2, 2)
    );
    assert!(compiler.compile_many(&a, &[]).is_err());
    assert_eq!(compiler.stats(), stats); // graph validation is not a backend miss
    compiler.clear();
    assert_eq!(compiler.stats().entries, 0);
    assert_eq!(compiler.stats().key_bytes, 0);
    assert_eq!(compiler.stats().hits, stats.hits);
    assert_eq!(compiler.stats().compile_time, stats.compile_time);
    drop(compiler);
    // Both eviction and compiler destruction release only their own references.
    assert_eq!(eb.run(&[&[1., 2.]]).unwrap(), [3., 4.]);
    assert_eq!(ec.run(&[&[1., 2.]]).unwrap(), [4., 5.]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cache_byte_limits_and_disable() {
    let client = client();
    let (g, y) = graph(1.);
    let bytes = g.stablehlo(&y).unwrap().len();
    let mut compiler = Compiler::new(
        client.clone(),
        CacheLimits {
            max_entries: 8,
            max_key_bytes: bytes,
        },
    );
    compiler.compile(&g, &y).unwrap();
    assert_eq!(compiler.stats().key_bytes, bytes);
    let (g2, y2) = graph(2.);
    compiler.compile(&g2, &y2).unwrap();
    assert_eq!(compiler.stats().evictions, 1);
    assert_eq!(compiler.stats().entries, 1);
    for limits in [
        CacheLimits {
            max_entries: 0,
            max_key_bytes: bytes,
        },
        CacheLimits {
            max_entries: 1,
            max_key_bytes: bytes - 1,
        },
    ] {
        let mut compiler = Compiler::new(client.clone(), limits);
        let a = compiler.compile(&g, &y).unwrap();
        let b = compiler.compile(&g, &y).unwrap();
        assert!(!Rc::ptr_eq(&a, &b));
        assert_eq!(compiler.stats().bypasses, 2);
        assert_eq!(compiler.stats().entries, 0);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cache_output_order_shape_and_client_isolation() {
    let client = client();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let y = x.add_scalar(1.).unwrap();
    let xy = compiler.compile_many(&g, &[x.clone(), y.clone()]).unwrap();
    let yx = compiler.compile_many(&g, &[y.clone(), x.clone()]).unwrap();
    assert!(!Rc::ptr_eq(&xy, &yx));
    assert_eq!(
        yx.run_many(&[&[2., 3.]]).unwrap(),
        [vec![3., 4.], vec![2., 3.]]
    );
    assert!(Rc::ptr_eq(
        &xy,
        &compiler.compile_many(&g, &[x, y]).unwrap()
    ));
    let (a, ya) = graph(1.);
    let original = compiler.compile(&a, &ya).unwrap();
    let mut other = Compiler::new(client, CacheLimits::default());
    assert!(!Rc::ptr_eq(&original, &other.compile(&a, &ya).unwrap()));
    let differently_shaped = Graph::default();
    let z = differently_shaped
        .input(&[1, 2])
        .unwrap()
        .add_scalar(1.)
        .unwrap();
    assert!(!Rc::ptr_eq(
        &original,
        &compiler.compile(&differently_shaped, &z).unwrap()
    ));
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_dead_branch_pruning_preserves_cache_and_input_abi() {
    let client = client();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let (g, y) = graph(1.);
    let first = compiler.compile(&g, &y).unwrap();
    let dead = y.mul_scalar(200.).unwrap().exp().unwrap();
    let same = compiler.compile(&g, &y).unwrap();
    assert!(Rc::ptr_eq(&first, &same));
    assert_eq!(same.run(&[&[2., 3.]]).unwrap(), [3., 4.]);
    // A previously pruned tensor remains a valid output of the original graph.
    let other = compiler.compile(&g, &dead).unwrap();
    assert!(!Rc::ptr_eq(&same, &other));
    assert_eq!(other.run(&[&[-1., -1.]]).unwrap(), [1., 1.]);
    let _index = g.input_i32_scalar().unwrap();
    let with_unused_input = compiler.compile(&g, &y).unwrap();
    assert!(!Rc::ptr_eq(&same, &with_unused_input));
    let x = client.buffer(&[2], &[5., 6.]).unwrap();
    let i = client.buffer(&[], &[99]).unwrap();
    assert!(with_unused_input.execute(&[&x]).is_err());
    assert_eq!(
        with_unused_input.execute(&[&x, &i]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [6., 7.]
    );
    assert_eq!((compiler.stats().hits, compiler.stats().misses), (1, 3));
}
#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_selective_invalidation_preserves_executables_and_lru() {
    use rxla_core::{CacheLimits, Client, Compiler, Graph};
    use std::rc::Rc;
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(
        client,
        CacheLimits {
            max_entries: 2,
            max_key_bytes: 1024 * 1024,
        },
    );
    let graph = Graph::default();
    let x = graph.input(&[1]).unwrap();
    let a = x.add_scalar(1.).unwrap();
    let b = x.add_scalar(2.).unwrap();
    let c = x.add_scalar(3.).unwrap();
    let program = graph.prepare(&a).unwrap();
    let held = compiler.compile(&graph, &a).unwrap();
    let other = compiler.compile(&graph, &b).unwrap();
    let before = compiler.stats();
    assert!(compiler.invalidate_memory(&program));
    assert!(!compiler.invalidate_memory(&program));
    assert_eq!(compiler.stats().entries, 1);
    assert!(compiler.stats().key_bytes < before.key_bytes);
    assert_eq!(compiler.stats().evictions, 0);
    assert_eq!(held.run(&[&[4.]]).unwrap(), [5.]);
    assert!(Rc::ptr_eq(&other, &compiler.compile(&graph, &b).unwrap()));
    let fresh = compiler.compile(&graph, &a).unwrap();
    assert!(!Rc::ptr_eq(&held, &fresh));
    compiler.compile(&graph, &c).unwrap(); // B, not A, is oldest.
    assert!(Rc::ptr_eq(&fresh, &compiler.compile(&graph, &a).unwrap()));
    compiler.compile(&graph, &b).unwrap();
    assert_eq!(compiler.stats().misses, 5);
    assert_eq!(compiler.stats().hits, 2);
    assert_eq!(compiler.stats().evictions, 2);
}
