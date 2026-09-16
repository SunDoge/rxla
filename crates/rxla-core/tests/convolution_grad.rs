use rxla_core::{Client, Conv2dOptions, Tracer};

fn client() -> Client {
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_convolution_both_gradients_match_independent_scalar_loops() {
    let client = client();
    let cases = [
        ([2, 4, 5, 2], [2, 3, 2, 3], Conv2dOptions::default()),
        (
            [2, 4, 5, 2],
            [2, 2, 2, 3],
            Conv2dOptions {
                strides: [2, 3],
                padding: [[1, 2], [2, 0]],
                dilation: [2, 1],
                ..Default::default()
            },
        ),
        (
            [1, 2, 3, 2],
            [1, 1, 2, 2],
            Conv2dOptions {
                strides: [3, 2],
                padding: [[4, 3], [2, 4]],
                ..Default::default()
            },
        ),
    ];
    for (shape, kernel_shape, options) in cases {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let w = g.input(&kernel_shape).unwrap();
        let y = x.conv2d(&w, options).unwrap();
        let output_shape = y.shape().to_vec();
        let seed = g.input(y.shape()).unwrap();
        let gradients = y.vjp(&[x, w], &seed).unwrap();
        let exe = g
            .compile_many(&client, &[y, gradients[0].clone(), gradients[1].clone()])
            .unwrap();
        let xn = shape.iter().product::<i64>() as usize;
        let wn = kernel_shape.iter().product::<i64>() as usize;
        let yn = output_shape.iter().product::<i64>() as usize;
        let seeds: Vec<_> = (0..yn).map(|i| ((i % 5) as f32 - 2.) * 0.25).collect();
        for scale in [1., -2.] {
            let values: Vec<_> = (0..xn)
                .map(|i| ((i % 13) as f32 - 6.) * 0.0625 * scale)
                .collect();
            let weights: Vec<_> = (0..wn)
                .map(|i| ((i % 7) as f32 - 3.) * 0.125 * scale)
                .collect();
            let mut expected_y = vec![0.0_f64; yn];
            let mut dx = vec![0.0_f64; xn];
            let mut dw = vec![0.0_f64; wn];
            let [batch, h, width, ci] = shape;
            let [kh, kw, ci_group, co] = kernel_shape;
            let oh = output_shape[1];
            let ow = output_shape[2];
            for n in 0..batch {
                for row in 0..oh {
                    for col in 0..ow {
                        for oc in 0..co {
                            let out = (((n * oh + row) * ow + col) * co + oc) as usize;
                            let group = oc / (co / options.groups);
                            for kr in 0..kh {
                                for kc in 0..kw {
                                    let ih = row * options.strides[0] + kr * options.dilation[0]
                                        - options.padding[0][0];
                                    let iw = col * options.strides[1] + kc * options.dilation[1]
                                        - options.padding[1][0];
                                    if ih < 0 || ih >= h || iw < 0 || iw >= width {
                                        continue;
                                    }
                                    for ic in 0..ci_group {
                                        let input = (((n * h + ih) * width + iw) * ci
                                            + group * ci_group
                                            + ic)
                                            as usize;
                                        let kernel =
                                            (((kr * kw + kc) * ci_group + ic) * co + oc) as usize;
                                        expected_y[out] +=
                                            f64::from(values[input]) * f64::from(weights[kernel]);
                                        dx[input] +=
                                            f64::from(seeds[out]) * f64::from(weights[kernel]);
                                        dw[kernel] +=
                                            f64::from(seeds[out]) * f64::from(values[input]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let actual = exe.run_many(&[&values, &weights, &seeds]).unwrap();
            for (output, expected) in [expected_y, dx, dw].iter().enumerate() {
                for (index, expected) in expected.iter().enumerate() {
                    assert!(
                        (f64::from(actual[output][index]) - expected).abs() < 1e-6,
                        "shape={shape:?} kernel={kernel_shape:?} options={options:?} output={output} index={index} actual={} expected={expected}",
                        actual[output][index]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_convolution_empty_inputs_and_outputs_have_zero_gradients() {
    let client = client();
    for (shape, kernel_shape, options) in [
        ([0, 3, 4, 2], [2, 2, 2, 3], Conv2dOptions::default()),
        (
            [1, 0, 2, 2],
            [1, 1, 2, 3],
            Conv2dOptions {
                padding: [[1, 1], [0, 0]],
                ..Default::default()
            },
        ),
        ([1, 1, 1, 2], [3, 3, 2, 3], Conv2dOptions::default()),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let w = g.input(&kernel_shape).unwrap();
        let loss = x
            .conv2d(&w, options)
            .unwrap()
            .sum(&[0, 1, 2, 3], false)
            .unwrap();
        let gradients = loss.grad(&[x, w]).unwrap();
        let exe = g.compile_many(&client, &gradients).unwrap();
        let xn = shape.iter().product::<i64>() as usize;
        let wn = kernel_shape.iter().product::<i64>() as usize;
        assert_eq!(
            exe.run_many(&[&vec![1.; xn], &vec![2.; wn]]).unwrap(),
            [vec![0.; xn], vec![0.; wn]]
        );
    }
}

#[test]
fn ungrouped_convolution_training_lowers_to_supported_ir() {
    let g = Tracer::default();
    let x = g.input(&[1, 3, 3, 1]).unwrap();
    let w = g.input(&[2, 2, 1, 1]).unwrap();
    let loss = x
        .conv2d(&w, Conv2dOptions::default())
        .unwrap()
        .sum(&[0, 1, 2, 3], false)
        .unwrap();
    let gradients = loss.grad(&[x, w]).unwrap();
    let lowered = g.prepare_many(&gradients).unwrap();
    let stablehlo = std::str::from_utf8(lowered.code()).unwrap();
    assert_eq!(stablehlo.matches("stablehlo.convolution").count(), 2);
    assert!(stablehlo.contains("[f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f]"));
    assert!(!stablehlo.contains("stablehlo.dot_general"));
}
