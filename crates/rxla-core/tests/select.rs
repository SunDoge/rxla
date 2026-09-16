use rxla_core::{Client, Tracer};

#[test]
fn selection_validates_every_operand() {
    let g = Tracer::default();
    let mask = g.input(&[2]).unwrap();
    let value = g.input(&[2]).unwrap();
    let wrong_shape = g.input(&[]).unwrap();
    let foreign = Tracer::default().input(&[2]).unwrap();
    for invalid in [wrong_shape, foreign] {
        assert!(mask.select(&invalid, &value).is_err());
        assert!(mask.select(&value, &invalid).is_err());
    }
    assert_eq!(mask.select(&value, &value).unwrap().shape(), [2]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_selection_nonfinite_values_and_dynamic_masks() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let mask = g.input(&[8]).unwrap();
    let a = g.input(&[8]).unwrap();
    let b = g.input(&[8]).unwrap();
    let selected = mask.select(&a, &b).unwrap();
    let finite_only = a.is_finite_mask().unwrap().select(&a, &b).unwrap();
    let exe = g.compile_many(&client, &[selected, finite_only]).unwrap();
    let av = [
        f32::NAN,
        f32::INFINITY,
        3.,
        -0.,
        5.,
        f32::NEG_INFINITY,
        7.,
        f32::NAN,
    ];
    let bv = [
        1.,
        2.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        6.,
        0.,
        8.,
    ];
    let ab = client.buffer(&[8], &av).unwrap();
    let bb = client.buffer(&[8], &bv).unwrap();
    for masks in [
        [0., -0., 1., -2., f32::INFINITY, 0., f32::NAN, 0.],
        [1.; 8],
        [0.; 8],
    ] {
        let mb = client.buffer(&[8], &masks).unwrap();
        let actual = exe.execute(&[&mb, &ab, &bb]).unwrap();
        for (output, conditions) in [masks.map(|m| m != 0.), av.map(|a| a.is_finite())]
            .into_iter()
            .enumerate()
        {
            for (i, value) in actual[output]
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let expected = if conditions[i] { av[i] } else { bv[i] };
                if expected.is_nan() {
                    assert!(value.is_nan());
                } else {
                    assert_eq!(value.to_bits(), expected.to_bits());
                }
            }
        }
    }
    for dims in [vec![], vec![0, 2]] {
        let g = Tracer::default();
        let mask = g.input(&dims).unwrap();
        let yes = g.constant(&[], &[3.]).unwrap().broadcast_to(&dims).unwrap();
        let no = g.constant(&[], &[7.]).unwrap().broadcast_to(&dims).unwrap();
        let exe = g
            .compile(&client, &mask.select(&yes, &no).unwrap())
            .unwrap();
        let values = if dims.is_empty() { vec![0.] } else { vec![] };
        let mb = client.buffer(&dims, &values).unwrap();
        assert_eq!(
            exe.execute(&[&mb]).unwrap()[0].to_vec::<f32>().unwrap(),
            vec![7.; values.len()]
        );
    }
}
