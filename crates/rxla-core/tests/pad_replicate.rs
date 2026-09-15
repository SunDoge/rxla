use rxla_core::{Client, Graph};

#[test]
fn replication_padding_preflight_and_empty_axis_rules() {
    let g = Graph::default();
    let x = g.input(&[2, 0]).unwrap();
    assert_eq!(x.pad_replicate(&[[1, 3], [0, 0]]).unwrap().shape(), [6, 0]);
    assert!(x.pad_replicate(&[[0, 0], [0, 1]]).is_err());
    assert!(x.pad_replicate(&[[0, 0]]).is_err());
    assert!(x.pad_replicate(&[[-1, 0], [0, 0]]).is_err());
    assert!(x.pad_replicate(&[[i64::MAX, 0], [0, 0]]).is_err());
    assert_eq!(
        g.input(&[]).unwrap().pad_replicate(&[]).unwrap().shape(),
        []
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_replication_values_and_edge_gradient_multiplicities() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, padding) in [
        (vec![3], vec![[2, 4]]),
        (vec![2, 3], vec![[1, 2], [2, 1]]),
        (vec![1, 2, 1], vec![[2, 3], [1, 0], [0, 2]]),
        (vec![2, 0], vec![[1, 2], [0, 0]]),
    ] {
        let g = Graph::default();
        let input = g.input(&shape).unwrap();
        let output = input.pad_replicate(&padding).unwrap();
        let loss = output
            .square()
            .unwrap()
            .sum(&(0..shape.len()).collect::<Vec<_>>(), false)
            .unwrap();
        let gradient = loss.grad(std::slice::from_ref(&input)).unwrap().remove(0);
        let hessian_sum = gradient
            .sum(&(0..shape.len()).collect::<Vec<_>>(), false)
            .unwrap()
            .grad(std::slice::from_ref(&input))
            .unwrap()
            .remove(0);
        let exe = g
            .compile_many(&client, &[output.clone(), gradient, hessian_sum])
            .unwrap();
        for offset in [0., 2.] {
            let values: Vec<_> = (0..shape.iter().product::<i64>())
                .map(|i| i as f32 - 1. + offset)
                .collect();
            let mut expected = Vec::new();
            let mut copies = vec![0.; values.len()];
            for flat in 0..output.shape().iter().product::<i64>() {
                let mut remainder = flat;
                let mut source = 0;
                let mut stride = 1;
                for axis in (0..shape.len()).rev() {
                    let coordinate = remainder % output.shape()[axis];
                    remainder /= output.shape()[axis];
                    let index = (coordinate - padding[axis][0]).clamp(0, shape[axis] - 1);
                    source += index * stride;
                    stride *= shape[axis];
                }
                expected.push(values[source as usize]);
                copies[source as usize] += 1.;
            }
            let actual = exe.run_many(&[&values]).unwrap();
            assert_eq!(actual[0], expected);
            assert_eq!(
                actual[1],
                values
                    .iter()
                    .zip(&copies)
                    .map(|(v, n)| 2. * v * n)
                    .collect::<Vec<_>>()
            );
            assert_eq!(actual[2], copies.iter().map(|n| 2. * n).collect::<Vec<_>>());
        }
    }
}
