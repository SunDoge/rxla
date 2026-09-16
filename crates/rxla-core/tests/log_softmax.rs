use rxla_core::{Client, Tracer};

#[test]
fn log_softmax_shape_validation() {
    let graph = Tracer::default();
    assert!(graph.input(&[]).unwrap().log_softmax(0).is_err());
    assert!(graph.input(&[2, 3]).unwrap().log_softmax(2).is_err());
    assert!(graph.input(&[2, 0]).unwrap().log_softmax(1).is_err());
    assert_eq!(
        graph
            .input(&[0, 3])
            .unwrap()
            .log_softmax(1)
            .unwrap()
            .shape(),
        [0, 3]
    );
}

// Independent F64 slice evaluator; supports arbitrary row-major reduction axes.
fn reference(values: &[f32], shape: &[usize], axis: usize) -> Vec<f64> {
    let stride: usize = shape[axis + 1..].iter().product();
    let width = shape[axis];
    (0..values.len())
        .map(|index| {
            let start = index / (width * stride) * width * stride + index % stride;
            let row: Vec<_> = (0..width)
                .map(|i| values[start + i * stride] as f64)
                .collect();
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = row.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln();
            values[index] as f64 - maximum - log_sum
        })
        .collect()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_log_softmax_matches_f64_on_every_axis() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let shape = [2, 3, 4];
    let values: Vec<f32> = (0..24)
        .map(|i| ((i * 17 % 29) as f32 - 14.) * 0.25)
        .collect();
    let shifted: Vec<_> = values.iter().map(|x| x + 10000.).collect();
    for axis in 0..3 {
        let graph = Tracer::default();
        let input = graph.input(&[2, 3, 4]).unwrap();
        let output = input.log_softmax(axis).unwrap();
        let executable = graph.compile(&client, &output).unwrap();
        for data in [&values, &shifted] {
            let actual = executable.run(&[data]).unwrap();
            for (actual, expected) in actual.iter().zip(reference(data, &shape, axis)) {
                assert!(
                    actual.is_finite() && (*actual as f64 - expected).abs() < 2e-6,
                    "axis {axis}: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_log_softmax_extremes_masks_and_empty_output() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let input = graph.input(&[2, 3]).unwrap();
    let output = input.log_softmax(1).unwrap();
    let executable = graph.compile(&client, &output).unwrap();
    let values = [1000., 0., -1000., 0., f32::NEG_INFINITY, 0.];
    let actual = executable.run(&[&values]).unwrap();
    assert_eq!(&actual[..3], &[0., -1000., -2000.]);
    assert_eq!(actual[4], f32::NEG_INFINITY);
    for index in [3, 5] {
        assert!((actual[index] + std::f32::consts::LN_2).abs() < 2e-6);
    }
    let graph = Tracer::default();
    let singleton = graph.input(&[2, 1]).unwrap().log_softmax(1).unwrap();
    assert_eq!(
        graph
            .compile(&client, &singleton)
            .unwrap()
            .run(&[&[10000., -10000.]])
            .unwrap(),
        [0., 0.]
    );
    let graph = Tracer::default();
    let empty = graph.input(&[0, 3]).unwrap().log_softmax(1).unwrap();
    assert!(
        graph
            .compile(&client, &empty)
            .unwrap()
            .run(&[&[]])
            .unwrap()
            .is_empty()
    );
}
