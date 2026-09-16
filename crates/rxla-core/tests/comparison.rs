use rxla_core::{Client, Error, Tensor, Tracer};

type Compare = fn(&Tensor, &Tensor) -> Result<Tensor, Error>;
const OPS: [Compare; 6] = [
    Tensor::eq_mask,
    Tensor::ne_mask,
    Tensor::lt_mask,
    Tensor::le_mask,
    Tensor::gt_mask,
    Tensor::ge_mask,
];

#[test]
fn comparison_shapes_and_ownership() {
    let g = Tracer::default();
    let a = g.input(&[2]).unwrap();
    let b = g.input(&[2]).unwrap();
    let scalar = g.input(&[]).unwrap();
    let foreign = Tracer::default().input(&[2]).unwrap();
    for op in OPS {
        assert_eq!(op(&a, &b).unwrap().shape(), [2]);
        assert!(op(&a, &scalar).is_err());
        assert!(op(&a, &foreign).is_err());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_comparison_ieee_semantics_and_selection() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let a = g.input(&[6, 6]).unwrap();
    let b = g.input(&[6, 6]).unwrap();
    let mut outputs: Vec<_> = OPS.iter().map(|op| op(&a, &b).unwrap()).collect();
    outputs.push(outputs[4].select(&a, &b).unwrap());
    let exe = g.compile_many(&client, &outputs).unwrap();
    for values in [
        [f32::NAN, f32::NEG_INFINITY, -0., 0., 1., f32::INFINITY],
        [f32::MIN, -123., -1., 1., 123., f32::MAX],
    ] {
        let av: Vec<_> = values.into_iter().flat_map(|v| [v; 6]).collect();
        let bv: Vec<_> = (0..6).flat_map(|_| values).collect();
        let ab = client.buffer(&[6, 6], &av).unwrap();
        let bb = client.buffer(&[6, 6], &bv).unwrap();
        let actual = exe.execute(&[&ab, &bb]).unwrap();
        for (op, output) in actual[..6].iter().enumerate() {
            let expected: Vec<f32> = av
                .iter()
                .zip(&bv)
                .map(|(&a, &b)| {
                    let comparisons = [a == b, a != b, a < b, a <= b, a > b, a >= b];
                    if comparisons[op] { 1. } else { 0. }
                })
                .collect();
            assert_eq!(output.to_vec::<f32>().unwrap(), expected);
        }
        for ((&a, &b), result) in av.iter().zip(&bv).zip(actual[6].to_vec::<f32>().unwrap()) {
            let expected = if a > b { a } else { b };
            if expected.is_nan() {
                assert!(result.is_nan());
            } else {
                assert_eq!(result.to_bits(), expected.to_bits());
            }
        }
    }
    for shape in [vec![], vec![0, 2]] {
        let g = Tracer::default();
        let a = g.input(&shape).unwrap();
        let b = g
            .constant(&[], &[0.])
            .unwrap()
            .broadcast_to(&shape)
            .unwrap();
        let outputs: Vec<_> = OPS.iter().map(|op| op(&a, &b).unwrap()).collect();
        let exe = g.compile_many(&client, &outputs).unwrap();
        let values = if shape.is_empty() { vec![0.] } else { vec![] };
        let input = client.buffer(&shape, &values).unwrap();
        for (output, expected) in exe
            .execute(&[&input])
            .unwrap()
            .iter()
            .zip([1., 0., 0., 1., 0., 1.])
        {
            assert_eq!(
                output.to_vec::<f32>().unwrap(),
                vec![expected; values.len()]
            );
        }
    }
}
