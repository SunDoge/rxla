use rxla_core::{Client, Tracer};

#[test]
fn f32_matmul_requests_highest_operand_precision() {
    let graph = Tracer::default();
    let a = graph.input(&[2, 3]).unwrap();
    let b = graph.input(&[3, 4]).unwrap();
    let output = a.matmul(&b).unwrap();
    let stablehlo = graph.stablehlo(&output).unwrap();
    assert_eq!(stablehlo.matches("stablehlo.dot_general").count(), 1);
    assert!(stablehlo.contains("precision = [HIGHEST, HIGHEST]"));
}

#[test]
fn matmul_shapes_and_output_validation() {
    let g = Tracer::default();
    for (lhs, rhs, expected) in [
        (vec![3], vec![3], vec![]),
        (vec![3], vec![2, 3, 4], vec![2, 4]),
        (vec![2, 4, 3], vec![3], vec![2, 4]),
        (vec![2, 1, 4, 3], vec![5, 3, 6], vec![2, 5, 4, 6]),
        (vec![0, 4, 3], vec![1, 3, 6], vec![0, 4, 6]),
    ] {
        let a = g.input(&lhs).unwrap();
        let b = g.input(&rhs).unwrap();
        assert_eq!(a.matmul(&b).unwrap().shape(), expected);
    }
    for (lhs, rhs) in [
        (vec![], vec![3]),
        (vec![2, 3], vec![4, 2]),
        (vec![2, 3, 4], vec![3, 4, 5]),
    ] {
        assert!(
            g.input(&lhs)
                .unwrap()
                .matmul(&g.input(&rhs).unwrap())
                .is_err()
        );
    }
    assert!(g.stablehlo_many(&[]).is_err());
    let foreign = Tracer::default().input(&[3]).unwrap();
    assert!(g.stablehlo_many(&[foreign]).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_broadcast_matmul_and_multiple_outputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let a = g.input(&[2, 1, 2, 3]).unwrap();
    let b = g.input(&[3, 3, 2]).unwrap();
    let y = a.matmul(&b).unwrap();
    let sum = y.sum(&[2, 3], false).unwrap();
    let exe = g.compile_many(&client, &[sum, y.clone(), y]).unwrap();
    assert!(matches!(
        exe.run(&[]).unwrap_err(),
        rxla_core::Error::InvalidArgument { message }
            if message.contains("multiple outputs")
    ));
    let av: Vec<f32> = (0..12).map(|v| v as f32 - 4.).collect();
    let bv: Vec<f32> = (0..18).map(|v| v as f32 / 2.).collect();
    let mut expected = Vec::new();
    for batch_a in 0..2 {
        for batch_b in 0..3 {
            for m in 0..2 {
                for n in 0..2 {
                    expected.push(
                        (0..3)
                            .map(|k| av[batch_a * 6 + m * 3 + k] * bv[batch_b * 6 + k * 2 + n])
                            .sum::<f32>(),
                    );
                }
            }
        }
    }
    let actual = exe.run_many(&[&av, &bv]).unwrap();
    assert_eq!(actual.len(), 3);
    assert_eq!(
        actual[0],
        expected
            .chunks(4)
            .map(|v| v.iter().sum::<f32>())
            .collect::<Vec<_>>()
    );
    assert_eq!(actual[1], expected);
    assert_eq!(actual[2], expected);
    let ab = client.buffer(a.shape(), &av).unwrap();
    let bb = client.buffer(b.shape(), &bv).unwrap();
    let resident = exe.execute(&[&ab, &bb]).unwrap();
    assert_eq!(resident[0].dimensions().unwrap(), [2, 3]);
    assert_eq!(resident[1].dimensions().unwrap(), [2, 3, 2, 2]);
    assert_eq!(resident[2].to_vec::<f32>().unwrap(), expected);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_vector_matmul() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let v = g.input(&[3]).unwrap();
    let m = g.constant(&[2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let dot = v.matmul(&v).unwrap();
    let mv = m.matmul(&v).unwrap();
    let vm = v.matmul(&m.transpose(&[1, 0]).unwrap()).unwrap();
    let exe = g.compile_many(&client, &[dot, mv, vm]).unwrap();
    assert_eq!(
        exe.run_many(&[&[1., 2., 3.]]).unwrap(),
        vec![vec![14.], vec![14., 32.], vec![14., 32.]]
    );
}
