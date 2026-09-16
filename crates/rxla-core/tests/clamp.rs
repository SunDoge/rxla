use rxla_core::{Client, Tracer};

#[test]
fn clamp_validation() {
    let g = Tracer::default();
    let x = g.input(&[4]).unwrap();
    assert!(x.clamp(2., 1.).is_err());
    assert!(x.clamp(f32::NAN, 1.).is_err());
    assert!(x.clamp(0., f32::NAN).is_err());
    assert!(x.hard_sigmoid(f32::INFINITY, 0.5).is_err());
    assert!(x.hard_sigmoid(0.2, f32::NAN).is_err());
    assert!(x.minimum(&g.input(&[1]).unwrap()).is_err());
    assert!(x.minimum(&Tracer::default().input(&[4]).unwrap()).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_clamp_and_exported_hard_sigmoid_variants() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[9]).unwrap();
    let values = [-10., -3., -2.5, -1., 0., 1., 2.5, 3., 10.];
    // The local OCR graph contains both slopes; they are not interchangeable.
    let outputs = [
        x.hard_sigmoid(0.2, 0.5).unwrap(),
        x.hard_sigmoid(0.166_666_7, 0.5).unwrap(),
        x.clamp(-1., 2.).unwrap(),
    ];
    let actual = g
        .compile_many(&client, &outputs)
        .unwrap()
        .run_many(&[&values])
        .unwrap();
    for (index, alpha) in [0.2f32, 0.166_666_7].into_iter().enumerate() {
        for (&x, &y) in values.iter().zip(&actual[index]) {
            let reference = (alpha as f64 * x as f64 + 0.5).clamp(0., 1.);
            assert!((y as f64 - reference).abs() < 2e-7);
        }
    }
    assert_eq!(actual[2], values.map(|v| v.clamp(-1., 2.)));
    let identity = x.clamp(f32::NEG_INFINITY, f32::INFINITY).unwrap();
    let values = [
        f32::NEG_INFINITY,
        -3.,
        -2.,
        -1.,
        0.,
        1.,
        2.,
        3.,
        f32::INFINITY,
    ];
    assert_eq!(
        g.compile(&client, &identity)
            .unwrap()
            .run(&[&values])
            .unwrap(),
        values
    );
}
