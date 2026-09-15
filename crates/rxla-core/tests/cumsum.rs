use rxla_core::{Client, Graph};

#[test]
fn cumsum_validates_axis_extent_and_compact_graph() {
    let g = Graph::default();
    assert!(g.input(&[]).unwrap().cumsum(0).is_err());
    assert!(g.input(&[2]).unwrap().cumsum(1).is_err());
    assert!(g.input(&[i64::MAX]).unwrap().cumsum(0).is_err());
    let counts: Vec<_> = [8, 4096]
        .into_iter()
        .map(|length| {
            let g = Graph::default();
            let y = g.input(&[length]).unwrap().cumsum(0).unwrap();
            let stablehlo = g.stablehlo(&y).unwrap();
            assert_eq!(stablehlo.matches("stablehlo.reduce_window").count(), 1);
            stablehlo.lines().count()
        })
        .collect();
    assert_eq!(counts[0], counts[1]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cumsum_all_axes_and_higher_gradients_match_scalar_reference() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [[2_i64, 3, 4], [2, 0, 4], [2, 1, 4], [1, 1, 1024]] {
        for axis in 0..3 {
            let g = Graph::default();
            let x = g.input(&shape).unwrap();
            let y = x.cumsum(axis).unwrap();
            let loss = y.mul(&y).unwrap().sum(&[0, 1, 2], false).unwrap();
            let dx = loss.grad(std::slice::from_ref(&x)).unwrap().remove(0);
            let ddx = dx
                .sum(&[0, 1, 2], false)
                .unwrap()
                .grad(std::slice::from_ref(&x))
                .unwrap()
                .remove(0);
            let exe = g.compile_many(&client, &[y, dx, ddx]).unwrap();
            let count = shape.iter().product::<i64>() as usize;
            let data: Vec<_> = (0..count).map(|i| (i as i32 % 7 - 3) as f32).collect();
            let input = client.buffer(&shape, &data).unwrap();
            let out = exe.execute(&[&input]).unwrap();
            let stride = shape[axis + 1..].iter().product::<i64>() as usize;
            let length = shape[axis] as usize;
            let prefix: Vec<_> = (0..count)
                .map(|i| {
                    let position = i / stride % length;
                    (0..=position)
                        .map(|p| data[i - (position - p) * stride])
                        .sum::<f32>()
                })
                .collect();
            let gradient: Vec<_> = (0..count)
                .map(|i| {
                    let position = i / stride % length;
                    2. * (position..length)
                        .map(|p| prefix[i + (p - position) * stride])
                        .sum::<f32>()
                })
                .collect();
            let second: Vec<_> = (0..count)
                .map(|i| {
                    let position = i / stride % length;
                    2. * (position..length).map(|p| (p + 1) as f32).sum::<f32>()
                })
                .collect();
            for (actual, expected) in out.iter().zip([prefix, gradient, second]) {
                assert_eq!(actual.dimensions().unwrap(), shape);
                assert_eq!(
                    actual.to_vec::<f32>().unwrap(),
                    expected,
                    "shape={shape:?} axis={axis}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cumsum_nonfinite_values_only_affect_their_prefix_successors() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[5]).unwrap();
    let exe = g.compile(&client, &x.cumsum(0).unwrap()).unwrap();
    let input = client
        .buffer(&[5], &[1., f32::INFINITY, 2., f32::NEG_INFINITY, f32::NAN])
        .unwrap();
    let result = exe.execute(&[&input]).unwrap()[0].to_vec::<f32>().unwrap();
    assert_eq!(result[..3], [1., f32::INFINITY, f32::INFINITY]);
    assert!(result[3..].iter().all(|x| x.is_nan()));
}
