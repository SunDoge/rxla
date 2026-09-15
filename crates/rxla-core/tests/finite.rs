use rxla_core::{Client, Graph};

#[test]
fn finite_mask_preserves_shape() {
    for shape in [vec![], vec![2, 3], vec![0, 3]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        assert_eq!(x.is_finite_mask().unwrap().shape(), shape);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_finite_mask_and_count() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[8]).unwrap();
    let mask = x.is_finite_mask().unwrap();
    let count = mask.sum(&[0], false).unwrap();
    let exe = g.compile_many(&client, &[mask, count]).unwrap();
    for values in [
        [
            0.,
            -0.,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            1.,
            -1.,
        ],
        [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.,
            1.,
            -1.,
            f32::from_bits(0xffc00001),
            f32::MAX,
        ],
        [f32::NAN; 8],
    ] {
        let input = client.buffer(&[8], &values).unwrap();
        let outputs = exe.execute(&[&input]).unwrap();
        let expected: Vec<f32> = values
            .iter()
            .map(|v| if v.is_finite() { 1. } else { 0. })
            .collect();
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), expected);
        assert_eq!(
            outputs[1].to_vec::<f32>().unwrap(),
            [expected.iter().sum::<f32>()]
        );
    }
    for shape in [vec![], vec![0, 3]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let exe = g.compile(&client, &x.is_finite_mask().unwrap()).unwrap();
        let values = if shape.is_empty() {
            vec![f32::INFINITY]
        } else {
            vec![]
        };
        let input = client.buffer(&shape, &values).unwrap();
        assert_eq!(
            exe.execute(&[&input]).unwrap()[0].to_vec::<f32>().unwrap(),
            vec![0.; values.len()]
        );
    }
}
