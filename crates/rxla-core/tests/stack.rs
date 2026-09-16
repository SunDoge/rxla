use rxla_core::{Client, Tensor, Tracer};

#[test]
fn stack_and_singleton_shape_validation() {
    let graph = Tracer::default();
    let x = graph.input(&[2, 1, 3]).unwrap();
    assert_eq!(x.squeeze(1).unwrap().shape(), [2, 3]);
    assert_eq!(x.unsqueeze(0).unwrap().shape(), [1, 2, 1, 3]);
    assert_eq!(x.unsqueeze(3).unwrap().shape(), [2, 1, 3, 1]);
    assert!(x.unsqueeze(4).is_err());
    assert!(x.squeeze(0).is_err());
    assert!(x.squeeze(3).is_err());
    assert!(Tensor::stack(&[], 0).is_err());
    assert!(Tensor::stack(std::slice::from_ref(&x), 4).is_err());
    assert!(Tensor::stack(&[x.clone(), graph.input(&[2, 3]).unwrap()], 0).is_err());
    assert!(Tensor::stack(&[x, Tracer::default().input(&[2, 1, 3]).unwrap()], 0).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_stack_preserves_values_on_every_insertion_axis() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [vec![], vec![3], vec![2, 3], vec![2, 0, 3]] {
        let count = shape.iter().product::<i64>() as usize;
        let a: Vec<_> = (0..count).map(|i| i as f32).collect();
        let b: Vec<_> = a.iter().map(|x| x + 10.).collect();
        for axis in 0..=shape.len() {
            let graph = Tracer::default();
            let x = graph.input(&shape).unwrap();
            let y = graph.input(&shape).unwrap();
            let joined = Tensor::stack(&[x.clone(), y], axis).unwrap();
            let singleton = Tensor::stack(std::slice::from_ref(&x), axis)
                .unwrap()
                .squeeze(axis)
                .unwrap();
            assert_eq!(singleton.shape(), shape);
            let mut expected_shape = shape.clone();
            expected_shape.insert(axis, 2);
            assert_eq!(joined.shape(), expected_shape);
            let outputs = graph
                .compile_many(&client, &[joined, singleton])
                .unwrap()
                .run_many(&[&a, &b])
                .unwrap();
            let chunk = shape[axis..].iter().product::<i64>() as usize;
            let mut expected = Vec::new();
            if chunk != 0 {
                for (a, b) in a.chunks(chunk).zip(b.chunks(chunk)) {
                    expected.extend_from_slice(a);
                    expected.extend_from_slice(b);
                }
            }
            assert_eq!(outputs[0], expected, "shape {shape:?}, axis {axis}");
            assert_eq!(outputs[1], a);
        }
    }
}
