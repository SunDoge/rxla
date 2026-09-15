use rxla_core::{Client, Conv2dOptions, Graph};

#[test]
fn forward_convolutions_use_highest_precision_and_training_is_explicitly_scoped() {
    let graph = Graph::default();
    let x = graph.input(&[1, 3, 3, 2]).unwrap();
    let w = graph.input(&[2, 2, 2, 3]).unwrap();
    let y = x.conv2d(&w, Default::default()).unwrap();
    assert!(y.sum(&[0, 1, 2, 3], false).unwrap().grad(&[x, w]).is_err());
    let transpose_w = graph.input(&[2, 2, 2, 3]).unwrap();
    let transposed = y
        .conv_transpose2d(&transpose_w, Default::default())
        .unwrap();
    let stablehlo = graph.stablehlo_many(&[y, transposed]).unwrap();
    assert_eq!(stablehlo.matches("stablehlo.convolution").count(), 2);
    assert_eq!(
        stablehlo
            .matches(
                "precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]"
            )
            .count(),
        2
    );
}

#[test]
fn convolution_validation() {
    let g = Graph::default();
    let x = g.input(&[1, 4, 5, 4]).unwrap();
    let k = g.input(&[3, 2, 2, 6]).unwrap();
    let options = Conv2dOptions {
        groups: 2,
        ..Default::default()
    };
    assert_eq!(x.conv2d(&k, options).unwrap().shape(), [1, 2, 4, 6]);
    for invalid in [
        Conv2dOptions {
            groups: 0,
            ..options
        },
        Conv2dOptions {
            groups: 3,
            ..options
        },
        Conv2dOptions {
            strides: [0, 1],
            ..options
        },
        Conv2dOptions {
            dilation: [1, -1],
            ..options
        },
        Conv2dOptions {
            padding: [[-1, 0], [0, 0]],
            ..options
        },
        Conv2dOptions {
            dilation: [i64::MAX, 1],
            ..options
        },
        Conv2dOptions {
            padding: [[i64::MAX, 0], [0, 0]],
            ..options
        },
    ] {
        assert!(x.conv2d(&k, invalid).is_err());
    }
    assert!(x.conv2d(&g.input(&[3, 2, 2, 5]).unwrap(), options).is_err());
    assert!(x.conv2d(&g.input(&[0, 2, 2, 6]).unwrap(), options).is_err());
    assert!(x.conv2d(&g.input(&[3, 2, 1, 6]).unwrap(), options).is_err());
    assert!(x.conv2d(&g.input(&[3, 2]).unwrap(), options).is_err());
    assert!(
        x.conv2d(&Graph::default().input(&[3, 2, 2, 6]).unwrap(), options)
            .is_err()
    );
}

// Scalar f64 cross-correlation reference; no tensor/BLAS/backend calls.
fn reference(
    x: &[f32],
    w: &[f32],
    xs: [i64; 4],
    ws: [i64; 4],
    ys: [i64; 4],
    o: Conv2dOptions,
) -> Vec<f64> {
    let mut result = Vec::new();
    for n in 0..ys[0] {
        for h in 0..ys[1] {
            for col in 0..ys[2] {
                for oc in 0..ys[3] {
                    let group = oc / (ws[3] / o.groups);
                    let mut sum = 0.;
                    for kh in 0..ws[0] {
                        for kw in 0..ws[1] {
                            let ih = h * o.strides[0] + kh * o.dilation[0] - o.padding[0][0];
                            let iw = col * o.strides[1] + kw * o.dilation[1] - o.padding[1][0];
                            if ih < 0 || ih >= xs[1] || iw < 0 || iw >= xs[2] {
                                continue;
                            }
                            for ic in 0..ws[2] {
                                let xi =
                                    ((n * xs[1] + ih) * xs[2] + iw) * xs[3] + group * ws[2] + ic;
                                let wi = ((kh * ws[1] + kw) * ws[2] + ic) * ws[3] + oc;
                                sum += x[xi as usize] as f64 * w[wi as usize] as f64;
                            }
                        }
                    }
                    result.push(sum);
                }
            }
        }
    }
    result
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_standard_grouped_depthwise_convolutions() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let cases = [
        (
            [1, 4, 5, 2],
            [3, 2, 2, 3],
            [1, 2, 4, 3],
            Conv2dOptions::default(),
        ),
        (
            [2, 6, 7, 4],
            [3, 2, 2, 6],
            [2, 2, 3, 6],
            Conv2dOptions {
                groups: 2,
                strides: [2, 3],
                dilation: [2, 1],
                padding: [[1, 0], [2, 1]],
            },
        ),
        (
            [1, 3, 4, 3],
            [2, 3, 1, 6],
            [1, 4, 4, 6],
            Conv2dOptions {
                groups: 3,
                padding: [[1, 1], [1, 1]],
                ..Default::default()
            },
        ),
        (
            [2, 2, 3, 4],
            [1, 1, 4, 5],
            [2, 2, 3, 5],
            Conv2dOptions::default(),
        ),
        (
            [1, 1, 1, 2],
            [3, 3, 2, 2],
            [1, 0, 0, 2],
            Conv2dOptions::default(),
        ),
    ];
    for (xs, ws, ys, options) in cases {
        let g = Graph::default();
        let x = g.input(&xs).unwrap();
        let w = g.input(&ws).unwrap();
        let y = x.conv2d(&w, options).unwrap();
        assert_eq!(y.shape(), ys);
        let activation = y.add_scalar(0.25).unwrap().silu().unwrap();
        let xv: Vec<f32> = (0..xs.iter().product::<i64>())
            .map(|i| ((i * 7 % 23) - 11) as f32 / 8.)
            .collect();
        let wv: Vec<f32> = (0..ws.iter().product::<i64>())
            .map(|i| ((i * 13 % 31) - 15) as f32 / 9.)
            .collect();
        let expected = reference(&xv, &wv, xs, ws, ys, options);
        let result = g
            .compile_many(&client, &[y, activation])
            .unwrap()
            .run_many(&[&xv, &wv])
            .unwrap();
        assert_eq!(result[0].len(), expected.len());
        assert_eq!(result[1].len(), expected.len());
        for (i, &value) in expected.iter().enumerate() {
            assert!((result[0][i] as f64 - value).abs() < 2e-5 + value.abs() * 1e-5);
            let activated = (value + 0.25) / (1. + (-(value + 0.25)).exp());
            assert!((result[1][i] as f64 - activated).abs() < 2e-5 + activated.abs() * 1e-5);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_small_vision_graph() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[1, 4, 4, 3]).unwrap();
    let w1 = g.input(&[3, 3, 3, 4]).unwrap();
    let w2 = g.input(&[3, 3, 1, 4]).unwrap();
    let head = g.input(&[4, 2]).unwrap();
    let o1 = Conv2dOptions {
        padding: [[1, 1]; 2],
        ..Default::default()
    };
    let o2 = Conv2dOptions {
        strides: [2, 2],
        groups: 4,
        ..o1
    };
    let features = x
        .conv2d(&w1, o1)
        .unwrap()
        .silu()
        .unwrap()
        .conv2d(&w2, o2)
        .unwrap()
        .silu()
        .unwrap();
    let probabilities = features
        .mean(&[1, 2], false)
        .unwrap()
        .matmul(&head)
        .unwrap()
        .softmax(1)
        .unwrap();
    let xv: Vec<f32> = (0..48).map(|i| (i as f32 - 24.) / 32.).collect();
    let a: Vec<f32> = (0..108).map(|i| ((i * 3 % 17) as f32 - 8.) / 16.).collect();
    let b: Vec<f32> = (0..36).map(|i| ((i * 7 % 13) as f32 - 6.) / 16.).collect();
    let h = [1., -1., -0.5, 0.5, 0.25, 1., -1., 0.5];
    let first: Vec<f32> = reference(&xv, &a, [1, 4, 4, 3], [3, 3, 3, 4], [1, 4, 4, 4], o1)
        .iter()
        .map(|&v| (v / (1. + (-v).exp())) as f32)
        .collect();
    let second: Vec<f64> = reference(&first, &b, [1, 4, 4, 4], [3, 3, 1, 4], [1, 2, 2, 4], o2)
        .iter()
        .map(|&v| v / (1. + (-v).exp()))
        .collect();
    let pooled: Vec<f64> = (0..4)
        .map(|c| (0..4).map(|p| second[p * 4 + c]).sum::<f64>() / 4.)
        .collect();
    let logits: Vec<f64> = (0..2)
        .map(|n| (0..4).map(|c| pooled[c] * h[c * 2 + n] as f64).sum())
        .collect();
    let p = 1. / (1. + (logits[1] - logits[0]).exp());
    let actual = g
        .compile(&client, &probabilities)
        .unwrap()
        .run(&[&xv, &a, &b, &h])
        .unwrap();
    assert_eq!(actual.len(), 2);
    assert!((actual[0] as f64 - p).abs() < 2e-6);
    assert!((actual[1] as f64 - (1. - p)).abs() < 2e-6);
}
