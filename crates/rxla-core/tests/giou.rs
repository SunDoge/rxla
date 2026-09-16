use rxla_core::{Client, Tracer};

fn reference(a: &[f64], b: &[f64]) -> f64 {
    let area = |v: &[f64]| (v[2] - v[0]).max(0.) * (v[3] - v[1]).max(0.);
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    let union = area(a) + area(b) - intersection;
    let enclosure = (a[2].max(b[2]) - a[0].min(b[0])) * (a[3].max(b[3]) - a[1].min(b[1]));
    let iou = if union > 0. { intersection / union } else { 0. };
    iou - if enclosure > 0. {
        (enclosure - union) / enclosure
    } else {
        0.
    }
}

#[test]
fn giou_validates_and_broadcasts_without_pairwise_axes() {
    let g = Tracer::default();
    let a = g.input(&[2, 1, 4]).unwrap();
    assert_eq!(
        a.box_giou(&g.input(&[3, 4]).unwrap()).unwrap().shape(),
        [2, 3]
    );
    assert!(a.box_giou(&g.input(&[2, 3]).unwrap()).is_err());
    assert!(a.box_giou(&g.input(&[3, 2, 4]).unwrap()).is_err());
    assert!(a.box_giou(&Tracer::default().input(&[4]).unwrap()).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_giou_values_and_disjoint_gradients_match_f64() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let a = g.input(&[4]).unwrap();
    let b = g.input(&[4]).unwrap();
    let y = a.box_giou(&b).unwrap();
    let mut roots = vec![y.clone()];
    roots.extend(y.grad(&[a, b]).unwrap());
    let exe = g.compile_many(&client, &roots).unwrap();
    for (av, bv, smooth) in [
        ([0., 0., 2., 3.], [0.5, 0.75, 4., 5.], true),
        ([0., 0., 2., 3.], [4., 5., 7., 9.], true),
        ([0., 0., 2., 3.], [-1., -2., 4., 5.], true),
        ([0., 0., 2., 3.], [0., 0., 2., 3.], false),
        ([0.; 4], [0.; 4], false),
    ] {
        let actual = exe.run_many(&[&av, &bv]).unwrap();
        let mut data = [av.map(f64::from), bv.map(f64::from)];
        assert!((actual[0][0] as f64 - reference(&data[0], &data[1])).abs() < 2e-7);
        assert!(actual.iter().flatten().all(|v| v.is_finite()));
        if smooth {
            for group in 0..2 {
                for index in 0..4 {
                    data[group][index] += 1e-5;
                    let plus = reference(&data[0], &data[1]);
                    data[group][index] -= 2e-5;
                    let minus = reference(&data[0], &data[1]);
                    data[group][index] += 1e-5;
                    assert!((actual[group + 1][index] as f64 - (plus - minus) / 2e-5).abs() < 1e-6);
                }
            }
        }
        if av[2] < bv[0] {
            assert!(actual[1].iter().any(|v| v.abs() > 1e-4));
            let moved: Vec<_> = av
                .iter()
                .zip(&actual[1])
                .map(|(&v, &gradient)| v + 0.1 * gradient)
                .collect();
            assert!(exe.run_many(&[&moved, &bv]).unwrap()[0][0] > actual[0][0]);
        }
    }
    let g = Tracer::default();
    let a = g.input(&[0, 4]).unwrap();
    let b = g.input(&[4]).unwrap();
    let y = a.box_giou(&b).unwrap();
    assert!(
        g.compile(&client, &y)
            .unwrap()
            .run(&[&[], &[0.; 4]])
            .unwrap()
            .is_empty()
    );
}
