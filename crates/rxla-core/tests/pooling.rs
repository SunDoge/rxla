use rxla_core::{Client, Graph, Pool2dOptions};

#[test]
fn pooling_shape_validation() {
    let g = Graph::default();
    let x = g.input(&[2, 5, 7, 3]).unwrap();
    assert_eq!(
        x.max_pool2d(Pool2dOptions::default()).unwrap().shape(),
        [2, 2, 3, 3]
    );
    for options in [
        Pool2dOptions {
            window: [0, 2],
            ..Default::default()
        },
        Pool2dOptions {
            strides: [-1, 1],
            ..Default::default()
        },
        Pool2dOptions {
            padding: [[-1, 0], [0, 0]],
            ..Default::default()
        },
        Pool2dOptions {
            padding: [[i64::MAX, 1], [0, 0]],
            ..Default::default()
        },
    ] {
        assert!(x.max_pool2d(options).is_err());
        assert!(x.avg_pool2d(options, true).is_err());
        assert!(x.avg_pool2d(options, false).is_err());
    }
    assert!(
        g.input(&[2, 3])
            .unwrap()
            .max_pool2d(Pool2dOptions::default())
            .is_err()
    );
    assert_eq!(
        x.max_pool2d(Pool2dOptions {
            window: [6, 8],
            ..Default::default()
        })
        .unwrap()
        .shape(),
        [2, 0, 0, 3]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_average_pooling_matches_f64_window_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2, 3, 4, 2];
    let values: Vec<f32> = (0..48).map(|i| ((i * 7 % 31) as f32 - 15.) / 8.).collect();
    for options in [
        Pool2dOptions::default(),
        Pool2dOptions {
            window: [3, 3],
            strides: [1, 1],
            padding: [[1, 1], [1, 1]],
        },
        Pool2dOptions {
            window: [2, 3],
            strides: [2, 1],
            padding: [[3, 0], [1, 2]],
        },
        Pool2dOptions {
            window: [1, 1],
            strides: [1, 1],
            padding: [[0, 0]; 2],
        },
        Pool2dOptions {
            window: [10, 10],
            ..Default::default()
        },
    ] {
        let graph = Graph::default();
        let x = graph.input(&shape).unwrap();
        let included = x.avg_pool2d(options, true).unwrap();
        let excluded = x.avg_pool2d(options, false).unwrap();
        let dims = included.shape().to_vec();
        let actual = graph
            .compile_many(&client, &[included, excluded])
            .unwrap()
            .run_many(&[&values])
            .unwrap();
        let mut index = 0;
        for n in 0..dims[0] {
            for oh in 0..dims[1] {
                for ow in 0..dims[2] {
                    for c in 0..dims[3] {
                        let mut sum = 0f64;
                        let mut count = 0;
                        for kh in 0..options.window[0] {
                            for kw in 0..options.window[1] {
                                let h = oh * options.strides[0] + kh - options.padding[0][0];
                                let w = ow * options.strides[1] + kw - options.padding[1][0];
                                if (0..shape[1]).contains(&h) && (0..shape[2]).contains(&w) {
                                    sum += values[(((n * shape[1] + h) * shape[2] + w) * shape[3]
                                        + c)
                                        as usize] as f64;
                                    count += 1;
                                }
                            }
                        }
                        for (result, denominator) in [
                            (&actual[0], options.window[0] * options.window[1]),
                            (&actual[1], count),
                        ] {
                            let expected = sum / denominator as f64;
                            if denominator == 0 {
                                assert!(result[index].is_nan());
                            } else {
                                assert!(
                                    result[index].is_finite()
                                        && (result[index] as f64 - expected).abs() < 2e-6
                                );
                            }
                        }
                        index += 1;
                    }
                }
            }
        }
        assert_eq!(actual[0].len(), index);
        assert_eq!(actual[1].len(), index);
    }
    for shape in [[0, 3, 4, 2], [2, 3, 4, 0]] {
        let graph = Graph::default();
        let x = graph.input(&shape).unwrap();
        let y = x.avg_pool2d(Pool2dOptions::default(), false).unwrap();
        assert!(
            graph
                .compile(&client, &y)
                .unwrap()
                .run(&[&[]])
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_pooling_matches_window_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2, 3, 4, 2];
    // All negative: zero-padding would give the wrong maximum on borders.
    let values: Vec<f32> = (0..48).map(|i| -1. - ((i * 7) % 31) as f32).collect();
    for options in [
        Pool2dOptions::default(),
        Pool2dOptions {
            window: [3, 3],
            strides: [1, 1],
            padding: [[1, 1], [1, 1]],
        },
        Pool2dOptions {
            window: [2, 3],
            strides: [2, 1],
            padding: [[3, 0], [1, 2]],
        },
        Pool2dOptions {
            window: [1, 1],
            strides: [1, 1],
            padding: [[0, 0]; 2],
        },
        Pool2dOptions {
            window: [10, 10],
            ..Default::default()
        },
    ] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let pooled = x.max_pool2d(options).unwrap();
        let dims = pooled.shape();
        let mut expected = Vec::new();
        for n in 0..dims[0] {
            for oh in 0..dims[1] {
                for ow in 0..dims[2] {
                    for c in 0..dims[3] {
                        let mut maximum = f32::NEG_INFINITY;
                        for kh in 0..options.window[0] {
                            for kw in 0..options.window[1] {
                                let h = oh * options.strides[0] + kh - options.padding[0][0];
                                let w = ow * options.strides[1] + kw - options.padding[1][0];
                                if (0..shape[1]).contains(&h) && (0..shape[2]).contains(&w) {
                                    let index = (((n * shape[1] + h) * shape[2] + w) * shape[3] + c)
                                        as usize;
                                    maximum = maximum.max(values[index]);
                                }
                            }
                        }
                        expected.push(maximum);
                    }
                }
            }
        }
        assert!(
            g.stablehlo(&pooled)
                .unwrap()
                .contains("stablehlo.reduce_window")
        );
        assert_eq!(
            g.compile(&client, &pooled)
                .unwrap()
                .run(&[&values])
                .unwrap(),
            expected
        );
    }
}
