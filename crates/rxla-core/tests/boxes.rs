use rxla_core::{Client, Tracer};

fn area(b: &[f64]) -> f64 {
    (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.)
}
fn iou(a: &[f64], b: &[f64]) -> f64 {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    let union = area(a) + area(b) - intersection;
    if union > 0. { intersection / union } else { 0. }
}

#[test]
fn box_shape_validation_and_batch_broadcast() {
    let g = Tracer::default();
    let a = g.input(&[2, 1, 3, 4]).unwrap();
    let b = g.input(&[5, 2, 4]).unwrap();
    assert_eq!(a.box_area().unwrap().shape(), [2, 1, 3]);
    assert_eq!(a.pairwise_box_iou(&b).unwrap().shape(), [2, 5, 3, 2]);
    for shape in [vec![], vec![3], vec![2, 5]] {
        let bad = g.input(&shape).unwrap();
        assert!(bad.box_area().is_err());
        assert!(a.pairwise_box_iou(&bad).is_err());
        assert!(a.box_iou(&bad).is_err());
    }
    assert!(g.input(&[4]).unwrap().pairwise_box_iou(&b).is_err());
    assert!(
        a.pairwise_box_iou(&Tracer::default().input(&[2, 4]).unwrap())
            .is_err()
    );
}

#[test]
fn aligned_iou_has_no_quadratic_intermediates() {
    let g = Tracer::default();
    let a = g.input(&[128, 4]).unwrap();
    let b = g.input(&[128, 4]).unwrap();
    let y = a.box_iou(&b).unwrap();
    assert_eq!(y.shape(), [128]);
    assert!(!g.stablehlo(&y).unwrap().contains("tensor<128x128"));
    assert!(a.box_iou(&g.input(&[2, 4]).unwrap()).is_err());
    assert!(a.box_iou(&Tracer::default().input(&[4]).unwrap()).is_err());
    assert_eq!(a.box_iou(&g.input(&[4]).unwrap()).unwrap().shape(), [128]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_pairwise_boxes_match_f64_and_smooth_gradients() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let boxes: Vec<f32> = vec![
        0., 0., 2., 3., 1., 1., 4., 5., 3., 2., 1., 4., 0., 0., 0., 0.,
    ];
    let g = Tracer::default();
    let a = g.input(&[4, 4]).unwrap();
    let b = g.input(&[4, 4]).unwrap();
    let output = a.pairwise_box_iou(&b).unwrap();
    let exe = g
        .compile_many(
            &client,
            &[output, a.box_area().unwrap(), a.box_iou(&b).unwrap()],
        )
        .unwrap();
    let actual = exe.run_many(&[&boxes, &boxes]).unwrap();
    assert_eq!(actual[2], [1., 1., 0., 0.]);
    let aligned_graph = Tracer::default();
    let aligned_a = aligned_graph.input(&[4, 4]).unwrap();
    let aligned_b = aligned_graph.input(&[4]).unwrap();
    let aligned_y = aligned_a.box_iou(&aligned_b).unwrap();
    let aligned_exe = aligned_graph.compile(&client, &aligned_y).unwrap();
    let aligned_values = aligned_exe.run(&[&boxes, &boxes[..4]]).unwrap();
    for i in 0..4 {
        assert_eq!(aligned_values[i], actual[0][i * 4]);
    }
    let batched = Tracer::default();
    let ba = batched.input(&[2, 1, 4]).unwrap();
    let bb = batched.input(&[1, 2, 4]).unwrap();
    let by = ba.pairwise_box_iou(&bb).unwrap();
    assert_eq!(by.shape(), [2, 1, 2]);
    let bexe = batched.compile(&client, &by).unwrap();
    let bv = bexe.run(&[&boxes[..8], &boxes[..8]]).unwrap();
    assert_eq!(bv, [actual[0][0], actual[0][1], actual[0][4], actual[0][5]]);
    let f64boxes: Vec<_> = boxes.iter().map(|&v| v as f64).collect();
    for i in 0..4 {
        assert_eq!(actual[1][i] as f64, area(&f64boxes[i * 4..i * 4 + 4]));
        for j in 0..4 {
            assert!(
                (actual[0][i * 4 + j] as f64
                    - iou(&f64boxes[i * 4..i * 4 + 4], &f64boxes[j * 4..j * 4 + 4]))
                .abs()
                    < 1e-7
            );
        }
    }
    // Strictly overlapping, unequal edges avoid nondifferentiable boundaries.
    let g = Tracer::default();
    let a = g.input(&[1, 4]).unwrap();
    let b = g.input(&[1, 4]).unwrap();
    let loss = a.pairwise_box_iou(&b).unwrap().sum(&[0, 1], false).unwrap();
    let mut grads = loss.grad(&[a.clone(), b.clone()]).unwrap();
    let aligned_loss = a.box_iou(&b).unwrap().sum(&[0], false).unwrap();
    grads.extend(aligned_loss.grad(&[a, b]).unwrap());
    let exe = g.compile_many(&client, &grads).unwrap();
    let av = [0., 0., 2., 3.];
    let bv = [0.5, 0.75, 4., 5.];
    let actual = exe.run_many(&[&av, &bv]).unwrap();
    assert_eq!(actual[..2], actual[2..]);
    let mut data = [av.map(f64::from), bv.map(f64::from)];
    for group in 0..2 {
        for index in 0..4 {
            data[group][index] += 1e-5;
            let plus = iou(&data[0], &data[1]);
            data[group][index] -= 2e-5;
            let minus = iou(&data[0], &data[1]);
            data[group][index] += 1e-5;
            assert!((actual[group][index] as f64 - (plus - minus) / 2e-5).abs() < 1e-6);
        }
    }
    for (n, m) in [(0, 2), (2, 0)] {
        let g = Tracer::default();
        let a = g.input(&[n, 4]).unwrap();
        let b = g.input(&[m, 4]).unwrap();
        let y = a.pairwise_box_iou(&b).unwrap();
        assert_eq!(y.shape(), [n, m]);
        let exe = g.compile(&client, &y).unwrap();
        assert!(
            exe.run(&[&vec![0.; n as usize * 4], &vec![0.; m as usize * 4]])
                .unwrap()
                .is_empty()
        );
    }
    let g = Tracer::default();
    let a = g.input(&[0, 4]).unwrap();
    let b = g.input(&[4]).unwrap();
    let y = a.box_iou(&b).unwrap();
    assert_eq!(y.shape(), [0]);
    assert!(
        g.compile(&client, &y)
            .unwrap()
            .run(&[&[], &boxes[..4]])
            .unwrap()
            .is_empty()
    );
}
