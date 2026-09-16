use rxla_core::{Client, Tracer};

#[test]
fn reverse_and_reflect_validate_axes_and_widths() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    assert!(x.flip(&[0, 0]).is_err());
    assert!(x.flip(&[2]).is_err());
    assert_eq!(x.flip(&[]).unwrap().shape(), [2, 3]);
    for padding in [
        vec![[2, 0], [0, 0]],
        vec![[0, 0], [0, 3]],
        vec![[-1, 0], [0, 0]],
        vec![],
    ] {
        assert!(x.pad_reflect(&padding).is_err());
    }
    assert!(g.input(&[1]).unwrap().pad_reflect(&[[1, 0]]).is_err());
    assert!(g.input(&[0]).unwrap().pad_reflect(&[[0, 1]]).is_err());
    assert_eq!(g.input(&[]).unwrap().pad_reflect(&[]).unwrap().shape(), []);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_flip_values_weighted_gradient_and_involution() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    let y = x.flip(&[1, 0]).unwrap();
    let weights = g.constant(&[2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
    let loss = y.mul(&weights).unwrap().sum(&[0, 1], false).unwrap();
    let gradient = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let restored = y.flip(&[0, 1]).unwrap();
    let exe = g.compile_many(&client, &[y, gradient, restored]).unwrap();
    let output = exe.run_many(&[&[10., 20., 30., 40., 50., 60.]]).unwrap();
    assert_eq!(output[0], [60., 50., 40., 30., 20., 10.]);
    assert_eq!(output[1], [6., 5., 4., 3., 2., 1.]);
    assert_eq!(output[2], [10., 20., 30., 40., 50., 60.]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_reflection_values_and_first_second_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, padding) in [
        (vec![3], vec![[1, 2]]),
        (vec![2, 3], vec![[1, 1], [2, 1]]),
        (vec![1, 3, 2], vec![[0, 0], [1, 2], [1, 1]]),
        (vec![0, 3], vec![[0, 0], [1, 2]]),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let y = x.pad_reflect(&padding).unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let loss = y.square().unwrap().sum(&axes, false).unwrap();
        let gradient = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
        let hessian_sum = gradient
            .sum(&axes, false)
            .unwrap()
            .grad(std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        let exe = g
            .compile_many(&client, &[y.clone(), gradient, hessian_sum])
            .unwrap();
        let values: Vec<_> = (0..shape.iter().product::<i64>())
            .map(|i| i as f32 - 2.)
            .collect();
        let mut expected = Vec::new();
        let mut copies = vec![0.; values.len()];
        for flat in 0..y.shape().iter().product::<i64>() {
            let mut rest = flat;
            let mut source = 0;
            let mut stride = 1;
            for axis in (0..shape.len()).rev() {
                let position = rest % y.shape()[axis] - padding[axis][0];
                rest /= y.shape()[axis];
                let coordinate = if position < 0 {
                    -position
                } else if position >= shape[axis] {
                    2 * shape[axis] - 2 - position
                } else {
                    position
                };
                source += coordinate * stride;
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
