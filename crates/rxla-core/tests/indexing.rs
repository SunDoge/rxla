use rxla_core::{Client, Tensor, Tracer};

#[test]
fn indexing_validation() {
    let g = Tracer::default();
    let x = g.input(&[3, 5]).unwrap();
    for (starts, limits, strides) in [
        (vec![0], vec![3], vec![1]),
        (vec![-1, 0], vec![3, 5], vec![1, 1]),
        (vec![2, 0], vec![1, 5], vec![1, 1]),
        (vec![0, 0], vec![3, 6], vec![1, 1]),
        (vec![0, 0], vec![3, 5], vec![0, 1]),
        (vec![0, 0], vec![3, 5], vec![1, -1]),
    ] {
        assert!(x.slice(&starts, &limits, &strides).is_err());
    }
    assert!(x.narrow(2, 0, 1).is_err());
    assert!(x.narrow(1, i64::MAX, 1).is_err());
    assert!(x.narrow(1, 0, -1).is_err());
    assert!(x.split(1, &[2, 2]).is_err());
    assert!(x.split(1, &[-1, 6]).is_err());
    assert!(x.split(1, &[i64::MAX, 6]).is_err());
    assert!(x.split(1, &[]).is_err());
    assert!(Tensor::concatenate(&[], 0).is_err());
    assert!(Tensor::concatenate(std::slice::from_ref(&x), 2).is_err());
    assert!(Tensor::concatenate(&[x.clone(), g.input(&[2, 5]).unwrap()], 1).is_err());
    assert!(
        Tensor::concatenate(&[x.clone(), Tracer::default().input(&[3, 5]).unwrap()], 1).is_err()
    );
    assert_eq!(x.slice(&[0, 0], &[3, 5], &[2, 3]).unwrap().shape(), [2, 2]);
    assert_eq!(x.narrow(1, 5, 0).unwrap().shape(), [3, 0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_slices_and_concatenation() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[3, 5]).unwrap();
    let pieces = x.split(1, &[2, 0, 3]).unwrap();
    let joined = Tensor::concatenate(&pieces, 1).unwrap();
    let reordered = Tensor::concatenate(&[pieces[2].clone(), pieces[0].clone()], 1).unwrap();
    let stride = x.slice(&[0, 1], &[3, 5], &[2, 2]).unwrap();
    let empty = x.narrow(0, 3, 0).unwrap();
    let exe = g
        .compile_many(&client, &[joined, reordered, stride, empty])
        .unwrap();
    let values: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let outputs = exe.run_many(&[&values]).unwrap();
    assert_eq!(outputs[0], values);
    assert_eq!(
        outputs[1],
        [
            2., 3., 4., 0., 1., 7., 8., 9., 5., 6., 12., 13., 14., 10., 11.
        ]
    );
    assert_eq!(outputs[2], [1., 3., 11., 13.]);
    assert!(outputs[3].is_empty());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_rotary_half_rotation() {
    // The rotate-half component of RoPE, shared by many Llama-style models.
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[1, 2, 4]).unwrap();
    let cos = g.input(&[1, 2, 4]).unwrap();
    let sin = g.input(&[1, 2, 4]).unwrap();
    let halves = x.split(2, &[2, 2]).unwrap();
    let rotated = Tensor::concatenate(&[halves[1].neg().unwrap(), halves[0].clone()], 2).unwrap();
    let y = x
        .mul(&cos)
        .unwrap()
        .add(&rotated.mul(&sin).unwrap())
        .unwrap();
    let xv = [1., 2., 3., 4., -1., -2., -3., -4.];
    let cv: Vec<f32> = [0.2f32, 0.7, 0.2, 0.7, 1.1, 1.8, 1.1, 1.8]
        .map(f32::cos)
        .into();
    let sv: Vec<f32> = [0.2f32, 0.7, 0.2, 0.7, 1.1, 1.8, 1.1, 1.8]
        .map(f32::sin)
        .into();
    let actual = g
        .compile(&client, &y)
        .unwrap()
        .run(&[&xv, &cv, &sv])
        .unwrap();
    for (i, &a) in actual.iter().enumerate() {
        let d = i % 4;
        let rotated = if d < 2 { -xv[i + 2] } else { xv[i - 2] };
        let expected = xv[i] * cv[i] + rotated * sv[i];
        assert!((a - expected).abs() < 1e-6);
    }
}
