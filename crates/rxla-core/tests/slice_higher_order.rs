use rxla_core::{Client, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_strided_slice_polynomial_derivatives_and_empty_selection() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for empty in [false, true] {
        let g = Graph::default();
        let x = g.input(&[3, 4]).unwrap();
        let selected = x
            .slice(&[1, 0], &[if empty { 1 } else { 3 }, 4], &[1, 2])
            .unwrap();
        let mut current = selected.square().unwrap().mul(&selected).unwrap();
        let mut roots = Vec::new();
        for _ in 0..4 {
            current = current
                .sum(&[0, 1], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            roots.push(current.clone());
        }
        let exe = g.compile_many(&client, &roots).unwrap();
        for scale in [1., -0.5] {
            let values: Vec<_> = (0..12).map(|i| (i as f32 - 3.) * scale).collect();
            let actual = exe.run_many(&[&values]).unwrap();
            for (i, x) in values.into_iter().enumerate() {
                let expected = if !empty && [4, 6, 8, 10].contains(&i) {
                    [3. * x * x, 6. * x, 6., 0.]
                } else {
                    [0.; 4]
                };
                for (order, expected) in expected.into_iter().enumerate() {
                    assert_eq!(actual[order][i], expected);
                }
            }
        }
    }
}
