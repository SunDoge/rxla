use rxla_core::{Client, Graph, RotaryLayout, Tensor};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_rotary_composite_gradients_match_independent_formula() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for layout in [RotaryLayout::SplitHalf, RotaryLayout::Interleaved] {
        let g = Graph::default();
        let x = g.input(&[2, 4]).unwrap();
        let cos = g.input(&[2]).unwrap();
        let sin = g.input(&[2]).unwrap();
        let weights = g.input(&[2, 4]).unwrap();
        let output = x.rotary_embedding(&cos, &sin, layout).unwrap();
        let grads = output
            .mul(&weights)
            .unwrap()
            .sum(&[0, 1], false)
            .unwrap()
            .grad(&[x, cos, sin])
            .unwrap();
        let exe = g.compile_many(&client, &grads).unwrap();
        let x = [1., -2., 3., 4., -1., 5., 2., -3.];
        let cos = [0.7, 0.4];
        let sin = [-0.2, 0.8];
        let weights = [0.5, 1., -0.5, 2., -1., 0.25, 0.75, -2.];
        let actual = exe.run_many(&[&x, &cos, &sin, &weights]).unwrap();
        let mut expected = [vec![0f64; 8], vec![0.; 2], vec![0.; 2]];
        for row in 0..2 {
            for pair in 0..2 {
                let (a, b) = match layout {
                    RotaryLayout::SplitHalf => (row * 4 + pair, row * 4 + pair + 2),
                    RotaryLayout::Interleaved => (row * 4 + pair * 2, row * 4 + pair * 2 + 1),
                };
                let (u, v) = (weights[a] as f64, weights[b] as f64);
                expected[0][a] = u * cos[pair] as f64 + v * sin[pair] as f64;
                expected[0][b] = -u * sin[pair] as f64 + v * cos[pair] as f64;
                expected[1][pair] += x[a] as f64 * u + x[b] as f64 * v;
                expected[2][pair] += -x[b] as f64 * u + x[a] as f64 * v;
            }
        }
        for (actual, expected) in actual.iter().zip(&expected) {
            for (&a, &e) in actual.iter().zip(expected) {
                assert!((a as f64 - e).abs() < 1e-6);
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_strided_slice_gradient_inserts_zeros() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (starts, limits, strides) in [
        ([1, 1], [5, 6], [2, 3]),
        ([0, 0], [5, 6], [2, 2]),
        ([2, 3], [2, 6], [1, 2]),
        ([4, 5], [5, 6], [99, 99]),
    ] {
        let g = Graph::default();
        let x = g.input(&[5, 6]).unwrap();
        let sliced = x.slice(&starts, &limits, &strides).unwrap();
        let weights = g.input(sliced.shape()).unwrap();
        let loss = sliced.mul(&weights).unwrap().sum(&[0, 1], false).unwrap();
        let grad = loss.grad(&[x]).unwrap();
        let exe = g.compile_many(&client, &grad).unwrap();
        let count = sliced.shape().iter().product::<i64>() as usize;
        let values: Vec<_> = (0..count).map(|i| i as f32 + 1.).collect();
        let mut expected = vec![0.; 30];
        let mut i = 0;
        for row in (starts[0]..limits[0]).step_by(strides[0] as usize) {
            for col in (starts[1]..limits[1]).step_by(strides[1] as usize) {
                expected[(row * 6 + col) as usize] = values[i];
                i += 1;
            }
        }
        assert_eq!(exe.run_many(&[&[1.; 30], &values]).unwrap(), [expected]);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_concat_pad_split_gradients_accumulate_shared_inputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for axis in [0, 1] {
        let g = Graph::default();
        let x = g.input(&[2, 3]).unwrap();
        let parts = x
            .split(axis, if axis == 0 { &[1, 0, 1] } else { &[1, 0, 2] })
            .unwrap();
        let rebuilt = Tensor::concatenate(&parts, axis).unwrap();
        let repeated = Tensor::concatenate(&[rebuilt, x.clone()], axis).unwrap();
        let padded = repeated.pad(&[[1, 2], [2, 1]], 9.).unwrap();
        let loss = padded.mul(&padded).unwrap().sum(&[0, 1], false).unwrap();
        let grad = loss.grad(&[x]).unwrap();
        let exe = g.compile_many(&client, &grad).unwrap();
        assert_eq!(
            exe.run_many(&[&[1., 2., 3., 4., 5., 6.]]).unwrap(),
            [vec![4., 8., 12., 16., 20., 24.]]
        );
    }
    for shape in [vec![], vec![0, 3]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let padded = x.pad(&vec![[1, 2]; shape.len()], 2.).unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let loss = padded.sum(&axes, false).unwrap();
        let grad = loss.grad(&[x]).unwrap();
        let input = vec![3.; usize::from(shape.is_empty())];
        assert_eq!(
            g.compile_many(&client, &grad)
                .unwrap()
                .run_many(&[&input])
                .unwrap(),
            [vec![1.; input.len()]]
        );
    }
}
