use rxla_core::{Client, Graph, Pool2dOptions};

fn client() -> Client {
    unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_max_pool_unique_winners_overlap_padding_and_stride() {
    let client = client();
    for (shape, options) in [
        (
            [2, 3, 4, 2],
            Pool2dOptions {
                window: [2, 3],
                strides: [1, 2],
                padding: [[1, 0], [0, 1]],
            },
        ),
        (
            [1, 4, 5, 2],
            Pool2dOptions {
                window: [2, 2],
                strides: [3, 3],
                padding: [[0, 0]; 2],
            },
        ),
        (
            [1, 2, 3, 2],
            Pool2dOptions {
                window: [2, 2],
                strides: [1, 2],
                padding: [[4, 3], [3, 4]],
            },
        ),
        (
            [1, 2, 2, 1],
            Pool2dOptions {
                window: [1, 1],
                strides: [5, 5],
                padding: [[2, 2]; 2],
            },
        ),
    ] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let y = x.max_pool2d(options).unwrap();
        let output_shape = y.shape().to_vec();
        let seed = g.input(y.shape()).unwrap();
        let dx = y.vjp(&[x], &seed).unwrap().remove(0);
        let exe = g.compile_many(&client, &[y, dx]).unwrap();
        let count = shape.iter().product::<i64>() as usize;
        let output_count = output_shape.iter().product::<i64>() as usize;
        let seeds: Vec<_> = (0..output_count)
            .map(|i| ((i % 7) as f32 - 3.) * 0.25)
            .collect();
        for sign in [1., -1.] {
            let values: Vec<_> = (0..count).map(|i| (i as f32 + 1.) * sign).collect();
            let mut expected_y = vec![f32::NEG_INFINITY; output_count];
            let mut expected_dx = vec![0.; count];
            let [n, h, w, channels] = shape;
            let oh = output_shape[1];
            let ow = output_shape[2];
            for batch in 0..n {
                for row in 0..oh {
                    for col in 0..ow {
                        for channel in 0..channels {
                            let output =
                                (((batch * oh + row) * ow + col) * channels + channel) as usize;
                            let mut winner = None;
                            for kh in 0..options.window[0] {
                                for kw in 0..options.window[1] {
                                    let ih = row * options.strides[0] - options.padding[0][0] + kh;
                                    let iw = col * options.strides[1] - options.padding[1][0] + kw;
                                    if ih >= 0 && ih < h && iw >= 0 && iw < w {
                                        let index = (((batch * h + ih) * w + iw) * channels
                                            + channel)
                                            as usize;
                                        if values[index] > expected_y[output] {
                                            expected_y[output] = values[index];
                                            winner = Some(index);
                                        }
                                    }
                                }
                            }
                            if let Some(index) = winner {
                                expected_dx[index] += seeds[output];
                            }
                        }
                    }
                }
            }
            let actual = exe.run_many(&[&values, &seeds]).unwrap();
            assert_eq!(
                actual,
                [expected_y, expected_dx],
                "shape={shape:?} options={options:?}"
            );
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_max_pool_ties_choose_one_winner_and_empty_gradients() {
    let client = client();
    let g = Graph::default();
    let x = g.input(&[1, 2, 2, 1]).unwrap();
    let y = x.max_pool2d(Pool2dOptions::default()).unwrap();
    let seed = g.constant(y.shape(), &[6.]).unwrap();
    let gradients = y.vjp(&[x], &seed).unwrap();
    let exe = g.compile_many(&client, &gradients).unwrap();
    for value in [1., 0., f32::NEG_INFINITY, f32::INFINITY] {
        let actual = exe.run_many(&[&[value; 4]]).unwrap();
        assert_eq!(actual[0].iter().filter(|&&v| v == 6.).count(), 1);
        assert_eq!(actual[0].iter().filter(|&&v| v == 0.).count(), 3);
    }
    for shape in [[0, 3, 4, 2], [2, 0, 3, 2], [1, 3, 4, 0], [1, 1, 1, 2]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let loss = x
            .max_pool2d(Pool2dOptions::default())
            .unwrap()
            .sum(&[0, 1, 2, 3], false)
            .unwrap();
        let gradients = loss.grad(&[x]).unwrap();
        let exe = g.compile_many(&client, &gradients).unwrap();
        let count = shape.iter().product::<i64>() as usize;
        assert_eq!(
            exe.run_many(&[&vec![1.; count]]).unwrap(),
            [vec![0.; count]]
        );
    }
}

#[test]
fn max_pool_higher_order_fails_explicitly() {
    let g = Graph::default();
    let x = g.input(&[1, 3, 3, 1]).unwrap();
    let loss = x
        .max_pool2d(Pool2dOptions::default())
        .unwrap()
        .sum(&[0, 1, 2, 3], false)
        .unwrap();
    assert!(loss.grad(std::slice::from_ref(&x)).is_err());
}
