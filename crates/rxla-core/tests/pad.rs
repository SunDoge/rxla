use rxla_core::{Client, Graph};

#[test]
fn padding_shape_validation_and_noop() {
    let g = Graph::default();
    let x = g.input(&[2, 3]).unwrap();
    assert_eq!(x.pad(&[[1, 2], [3, 4]], 0.).unwrap().shape(), [5, 10]);
    assert!(x.pad(&[[1, 2]], 0.).is_err());
    assert!(x.pad(&[[-1, 0], [0, 0]], 0.).is_err());
    assert!(x.pad(&[[i64::MAX, 0], [0, 0]], 0.).is_err());
    assert!(x.pad(&[[i64::MAX / 2, 0], [i64::MAX / 2, 0]], 0.).is_err());
    let noop = x.pad(&[[0, 0]; 2], f32::NAN).unwrap();
    assert_eq!(g.stablehlo(&x).unwrap(), g.stablehlo(&noop).unwrap());
    assert_eq!(g.input(&[]).unwrap().pad(&[], 0.).unwrap().shape(), []);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_padding_matches_row_major_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for (shape, padding) in [
        (vec![3], vec![[2, 1]]),
        (vec![2, 3], vec![[1, 0], [2, 1]]),
        (vec![2, 1, 3], vec![[0, 1], [2, 0], [1, 2]]),
        (vec![0, 2], vec![[1, 2], [1, 1]]),
        (vec![0, 2], vec![[0, 0], [1, 1]]),
    ] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let output = x.pad(&padding, -7.5).unwrap();
        let dims = output.shape().to_vec();
        let exe = g.compile(&client, &output).unwrap();
        for offset in [0., 100.] {
            let values: Vec<_> = (0..shape.iter().product::<i64>())
                .map(|i| i as f32 + offset)
                .collect();
            let actual = exe.run(&[&values]).unwrap();
            let mut expected = Vec::new();
            for flat in 0..dims.iter().product::<i64>() {
                let mut rest = flat;
                let mut source = 0;
                let mut stride = 1;
                let mut inside = true;
                for axis in (0..shape.len()).rev() {
                    let coordinate = rest % dims[axis] - padding[axis][0];
                    rest /= dims[axis];
                    inside &= (0..shape[axis]).contains(&coordinate);
                    source += coordinate * stride;
                    stride *= shape[axis];
                }
                expected.push(if inside {
                    values[source as usize]
                } else {
                    -7.5
                });
            }
            assert_eq!(actual, expected);
        }
    }
    let g = Graph::default();
    let x = g.input(&[1]).unwrap();
    let nan_pad = x.pad(&[[1, 1]], f32::NAN).unwrap();
    let inf_pad = x.pad(&[[1, 1]], f32::NEG_INFINITY).unwrap();
    let outputs = g
        .compile_many(&client, &[nan_pad, inf_pad])
        .unwrap()
        .run_many(&[&[3.]])
        .unwrap();
    assert!(outputs[0][0].is_nan() && outputs[0][2].is_nan());
    assert_eq!(outputs[0][1], 3.);
    assert_eq!(outputs[1], [f32::NEG_INFINITY, 3., f32::NEG_INFINITY]);
}
