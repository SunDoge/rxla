use rxla_core::{Client, ConvTranspose2dOptions, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_transposed_convolution_gradients_match_scalar_scatter() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, kernel_shape, options) in [
        (
            [2, 2, 3, 2],
            [2, 3, 3, 2],
            ConvTranspose2dOptions::default(),
        ),
        (
            [1, 3, 2, 2],
            [2, 2, 3, 2],
            ConvTranspose2dOptions {
                strides: [2, 3],
                padding: [[1, 0], [0, 2]],
                dilation: [2, 1],
                output_padding: [1, 2],
            },
        ),
        (
            [1, 2, 2, 1],
            [1, 1, 2, 1],
            ConvTranspose2dOptions {
                strides: [3, 2],
                output_padding: [2, 1],
                ..Default::default()
            },
        ),
        (
            [0, 2, 3, 2],
            [2, 3, 3, 2],
            ConvTranspose2dOptions::default(),
        ),
    ] {
        let g = Tracer::default();
        let x = g.input(&shape).unwrap();
        let w = g.input(&kernel_shape).unwrap();
        let y = x.conv_transpose2d(&w, options).unwrap();
        let output_shape = y.shape().to_vec();
        let seed = g.input(y.shape()).unwrap();
        let gradients = y.vjp(&[x, w], &seed).unwrap();
        let exe = g
            .compile_many(&client, &[y, gradients[0].clone(), gradients[1].clone()])
            .unwrap();
        let xn = shape.iter().product::<i64>() as usize;
        let wn = kernel_shape.iter().product::<i64>() as usize;
        let yn = output_shape.iter().product::<i64>() as usize;
        let seeds: Vec<_> = (0..yn).map(|i| ((i % 7) as f32 - 3.) * 0.25).collect();
        for scale in [1., -2.] {
            let values: Vec<_> = (0..xn)
                .map(|i| ((i % 11) as f32 - 5.) * 0.125 * scale)
                .collect();
            let weights: Vec<_> = (0..wn)
                .map(|i| ((i % 5) as f32 - 2.) * 0.0625 * scale)
                .collect();
            let mut expected_y = vec![0.0_f64; yn];
            let mut dx = vec![0.0_f64; xn];
            let mut dw = vec![0.0_f64; wn];
            let [n, h, width, ci] = shape;
            let [kh, kw, co, _] = kernel_shape;
            let oh = output_shape[1];
            let ow = output_shape[2];
            for batch in 0..n {
                for row in 0..h {
                    for col in 0..width {
                        for ic in 0..ci {
                            let input = (((batch * h + row) * width + col) * ci + ic) as usize;
                            for kr in 0..kh {
                                for kc in 0..kw {
                                    let out_h = row * options.strides[0] + kr * options.dilation[0]
                                        - options.padding[0][0];
                                    let out_w = col * options.strides[1] + kc * options.dilation[1]
                                        - options.padding[1][0];
                                    if out_h < 0 || out_h >= oh || out_w < 0 || out_w >= ow {
                                        continue;
                                    }
                                    for oc in 0..co {
                                        let output = (((batch * oh + out_h) * ow + out_w) * co + oc)
                                            as usize;
                                        let kernel =
                                            (((kr * kw + kc) * co + oc) * ci + ic) as usize;
                                        expected_y[output] +=
                                            f64::from(values[input]) * f64::from(weights[kernel]);
                                        dx[input] +=
                                            f64::from(seeds[output]) * f64::from(weights[kernel]);
                                        dw[kernel] +=
                                            f64::from(seeds[output]) * f64::from(values[input]);
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
                        "options={options:?} output={output} index={index}"
                    );
                }
            }
        }
    }
}

#[test]
fn transposed_kernel_gradient_is_outside_the_supported_training_surface() {
    let g = Tracer::default();
    let x = g.input(&[1, 2, 2, 1]).unwrap();
    let w = g.input(&[2, 2, 1, 1]).unwrap();
    let loss = x
        .conv_transpose2d(&w, Default::default())
        .unwrap()
        .sum(&[0, 1, 2, 3], false)
        .unwrap();
    let error = loss
        .grad(std::slice::from_ref(&w))
        .err()
        .expect("transposed convolution training is intentionally unsupported");
    assert!(error.to_string().contains("Conv2dKernelGradient"));
}
