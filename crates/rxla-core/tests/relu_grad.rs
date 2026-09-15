use rxla_core::{Client, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_relu_forward_and_zero_point_gradient() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[6]).unwrap();
    let weights = g.input(&[6]).unwrap();
    let y = x.relu().unwrap();
    let dy = y
        .mul(&weights)
        .unwrap()
        .sum(&[0], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g.compile_many(&client, &[y, dy]).unwrap();
    assert_eq!(
        exe.run_many(&[&[-2., -0., 0., 0.01, 1., 100.], &[1., 2., 3., 4., 5., 6.]])
            .unwrap(),
        [
            vec![0., 0., 0., 0.01, 1., 100.],
            vec![0., 0., 0., 4., 5., 6.]
        ]
    );
    let g = Graph::default();
    let x = g.input(&[3]).unwrap();
    let exe = g.compile(&client, &x.relu().unwrap()).unwrap();
    let values = exe
        .run(&[&[f32::NEG_INFINITY, f32::INFINITY, f32::NAN]])
        .unwrap();
    assert_eq!(values[..2], [0., f32::INFINITY]);
    assert!(values[2].is_nan());
    for shape in [vec![], vec![0, 2]] {
        let g = Graph::default();
        let x = g.input(&shape).unwrap();
        let y = x.relu().unwrap();
        let axes: Vec<_> = (0..shape.len()).collect();
        let grad = y.sum(&axes, false).unwrap().grad(&[x]).unwrap();
        let exe = g.compile_many(&client, &grad).unwrap();
        let input = vec![2.; usize::from(shape.is_empty())];
        assert_eq!(exe.run_many(&[&input]).unwrap(), [vec![1.; input.len()]]);
    }
}
