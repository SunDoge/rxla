use rxla_core::{Client, Graph, Pool2dOptions};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_average_pool_grad_matches_scalar_scatter_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let cases = [
        (
            [2, 3, 4, 3],
            Pool2dOptions {
                window: [2, 3],
                strides: [1, 2],
                padding: [[1, 0], [0, 1]],
            },
            false,
        ),
        (
            [2, 3, 4, 3],
            Pool2dOptions {
                window: [2, 3],
                strides: [1, 2],
                padding: [[1, 0], [0, 1]],
            },
            true,
        ),
        (
            [1, 4, 5, 2],
            Pool2dOptions {
                window: [2, 2],
                strides: [3, 3],
                padding: [[0, 0]; 2],
            },
            false,
        ),
        (
            [1, 3, 4, 2],
            Pool2dOptions {
                window: [1, 1],
                strides: [2, 3],
                padding: [[0, 0]; 2],
            },
            true,
        ),
        (
            [1, 2, 3, 2],
            Pool2dOptions {
                window: [2, 2],
                strides: [1, 2],
                padding: [[4, 3], [3, 4]],
            },
            true,
        ),
    ];
    for (shape, options, include_pad) in cases {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let y = x.avg_pool2d(options, include_pad).unwrap();
        let seed = g.input(y.shape()).unwrap();
        let dx = y.vjp(std::slice::from_ref(&x), &seed).unwrap().remove(0);
        let nonlinear = x
            .square()
            .unwrap()
            .avg_pool2d(options, include_pad)
            .unwrap()
            .vjp(&[x], &seed)
            .unwrap()
            .remove(0);
        let output_shape = y.shape().to_vec();
        let exe = g.compile_many(&client, &[y, dx, nonlinear]).unwrap();
        let count = shape.iter().product::<i64>() as usize;
        let output_count = output_shape.iter().product::<i64>() as usize;
        let seeds: Vec<_> = (0..output_count)
            .map(|i| ((i % 7) as f32 - 3.) * 0.25)
            .collect();
        for scale in [1., -2.] {
            let values: Vec<_> = (0..count)
                .map(|i| ((i % 11) as f32 - 5.) * 0.125 * scale)
                .collect();
            let mut expected_y = vec![0.0_f64; output_count];
            let mut expected_dx = vec![0.0_f64; count];
            let [n, h, w, channels] = shape;
            let oh = output_shape[1];
            let ow = output_shape[2];
            for batch in 0..n {
                for row in 0..oh {
                    for col in 0..ow {
                        for channel in 0..channels {
                            let output =
                                (((batch * oh + row) * ow + col) * channels + channel) as usize;
                            let mut indices = Vec::new();
                            for kh in 0..options.window[0] {
                                for kw in 0..options.window[1] {
                                    let ih = row * options.strides[0] - options.padding[0][0] + kh;
                                    let iw = col * options.strides[1] - options.padding[1][0] + kw;
                                    if ih >= 0 && ih < h && iw >= 0 && iw < w {
                                        indices.push(
                                            (((batch * h + ih) * w + iw) * channels + channel)
                                                as usize,
                                        );
                                    }
                                }
                            }
                            let divisor = if include_pad {
                                (options.window[0] * options.window[1]) as f64
                            } else {
                                indices.len() as f64
                            };
                            for index in indices {
                                expected_y[output] += f64::from(values[index]) / divisor;
                                expected_dx[index] += f64::from(seeds[output]) / divisor;
                            }
                        }
                    }
                }
            }
            let actual = exe.run_many(&[&values, &seeds]).unwrap();
            for (actual, expected) in actual[0].iter().zip(expected_y) {
                assert!((f64::from(*actual) - expected).abs() < 1e-6);
            }
            for (i, expected) in expected_dx.into_iter().enumerate() {
                assert!(
                    (f64::from(actual[1][i]) - expected).abs() < 1e-6,
                    "shape={shape:?} options={options:?} index={i}"
                );
                assert!(
                    (f64::from(actual[2][i]) - 2. * f64::from(values[i]) * expected).abs() < 1e-6
                );
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_average_pool_empty_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [[0, 3, 4, 2], [2, 0, 3, 2], [1, 3, 4, 0], [1, 1, 1, 2]] {
        for include_pad in [false, true] {
            let g = Graph::default();
            let x = g.input(&shape).unwrap();
            let y = x.avg_pool2d(Pool2dOptions::default(), include_pad).unwrap();
            let gradient = y.sum(&[0, 1, 2, 3], false).unwrap().grad(&[x]).unwrap();
            let exe = g.compile_many(&client, &gradient).unwrap();
            let count = shape.iter().product::<i64>() as usize;
            assert_eq!(
                exe.run_many(&[&vec![1.; count]]).unwrap(),
                [vec![0.; count]]
            );
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_average_pool_squared_input_second_derivative() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[1, 3, 3, 1]).unwrap();
    let loss = x
        .square()
        .unwrap()
        .avg_pool2d(Pool2dOptions::default(), true)
        .unwrap()
        .sum(&[0, 1, 2, 3], false)
        .unwrap();
    let first = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
    let second = first
        .sum(&[0, 1, 2, 3], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[first, second]).unwrap();
    assert_eq!(
        exe.run_many(&[&[2.; 9]]).unwrap(),
        [
            vec![1., 1., 0., 1., 1., 0., 0., 0., 0.],
            vec![0.5, 0.5, 0., 0.5, 0.5, 0., 0., 0., 0.],
        ]
    );
}
