use rxla_core::{Client, RotaryLayout, Tracer};

#[test]
fn rotary_shape_and_ownership_validation() {
    let g = Tracer::default();
    let c = g.input(&[2]).unwrap();
    for shape in [&[][..], &[0], &[3]] {
        assert!(
            g.input(shape)
                .unwrap()
                .rotary_embedding(&c, &c, RotaryLayout::SplitHalf)
                .is_err()
        );
    }
    let x = g.input(&[2, 3, 4]).unwrap();
    let full = g.input(&[4]).unwrap();
    assert!(
        x.rotary_embedding(&full, &c, RotaryLayout::SplitHalf)
            .is_err()
    );
    let foreign = Tracer::default().input(&[2]).unwrap();
    assert!(
        x.rotary_embedding(&c, &foreign, RotaryLayout::SplitHalf)
            .is_err()
    );
    for layout in [RotaryLayout::SplitHalf, RotaryLayout::Interleaved] {
        assert_eq!(
            x.rotary_embedding(&c, &c, layout).unwrap().shape(),
            x.shape()
        );
        let empty = g.input(&[0, 4]).unwrap();
        assert_eq!(
            empty.rotary_embedding(&c, &c, layout).unwrap().shape(),
            [0, 4]
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_rotary_layouts_match_f64_with_runtime_angles() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[2, 3, 4]).unwrap();
    let cos = g.input(&[3, 2]).unwrap();
    let sin = g.input(&[3, 2]).unwrap();
    let outputs = [RotaryLayout::SplitHalf, RotaryLayout::Interleaved]
        .map(|layout| x.rotary_embedding(&cos, &sin, layout).unwrap());
    let exe = g.compile_many(&client, &outputs).unwrap();
    let values: Vec<f32> = (0..24).map(|i| (i as f32 - 12.) / 3.).collect();
    for offset in [0f32, 0.7] {
        let angles: Vec<f32> = (0..6).map(|i| i as f32 / 5. + offset).collect();
        let cos: Vec<f32> = angles.iter().map(|v| v.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|v| v.sin()).collect();
        let actual = exe.run_many(&[&values, &cos, &sin]).unwrap();
        for (layout, result) in actual.iter().enumerate() {
            for row in 0..6 {
                for pair in 0..2 {
                    let (a, b) = if layout == 0 {
                        (pair, pair + 2)
                    } else {
                        (pair * 2, pair * 2 + 1)
                    };
                    let c = cos[(row % 3) * 2 + pair] as f64;
                    let s = sin[(row % 3) * 2 + pair] as f64;
                    let av = values[row * 4 + a] as f64;
                    let bv = values[row * 4 + b] as f64;
                    assert!((result[row * 4 + a] as f64 - (av * c - bv * s)).abs() < 1e-6);
                    assert!((result[row * 4 + b] as f64 - (bv * c + av * s)).abs() < 1e-6);
                }
            }
        }
    }
}
