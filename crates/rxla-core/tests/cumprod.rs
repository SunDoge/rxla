use rxla_core::{Client, Tracer};

#[test]
fn validates_axis_and_lowers_multiplication_work() {
    let g = Tracer::default();
    assert!(g.input(&[]).unwrap().cumprod(0).is_err());
    assert!(g.input(&[2]).unwrap().cumprod(1).is_err());
    for length in [8, 4096] {
        let g = Tracer::default();
        let y = g.input(&[length]).unwrap().cumprod(0).unwrap();
        let stablehlo = g.stablehlo(&y).unwrap();
        assert!(stablehlo.contains("stablehlo.multiply"));
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_products_zero_safe_vjp_and_hessian_match_polynomials() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    for shape in [[2_i64, 5], [3, 4], [2, 0], [2, 1]] {
        for axis in 0..2 {
            let count = shape.iter().product::<i64>() as usize;
            let data: Vec<_> = (0..count)
                .map(|i| [0_f32, 2., -1., 0., 0.5][i % 5])
                .collect();
            let seeds: Vec<_> = (0..count).map(|i| [0.5_f32, -1., 2.][i % 3]).collect();
            let direction: Vec<_> = (0..count).map(|i| (i % 3) as f32 - 1.).collect();
            let stride = shape[axis + 1..].iter().product::<i64>() as usize;
            let length = shape[axis] as usize;
            let mut prefix = vec![0_f64; count];
            let mut gradient = vec![0_f64; count];
            let mut hvp = vec![0_f64; count];
            for end_index in 0..count {
                let end = end_index / stride % length;
                let indices: Vec<_> = (0..=end).map(|p| end_index - (end - p) * stride).collect();
                prefix[end_index] = indices.iter().map(|&i| data[i] as f64).product();
                for &i in &indices {
                    gradient[i] += seeds[end_index] as f64
                        * indices
                            .iter()
                            .filter(|&&k| k != i)
                            .map(|&k| data[k] as f64)
                            .product::<f64>();
                    for &j in &indices {
                        if i != j {
                            hvp[i] += seeds[end_index] as f64
                                * direction[j] as f64
                                * indices
                                    .iter()
                                    .filter(|&&k| k != i && k != j)
                                    .map(|&k| data[k] as f64)
                                    .product::<f64>();
                        }
                    }
                }
            }
            let g = Tracer::default();
            let x = g.input(&shape).unwrap();
            let seed = g.input(&shape).unwrap();
            let vector = g.input(&shape).unwrap();
            let y = x.cumprod(axis).unwrap();
            let dx = y.vjp(std::slice::from_ref(&x), &seed).unwrap().remove(0);
            let h = dx
                .mul(&vector)
                .unwrap()
                .sum(&[0, 1], false)
                .unwrap()
                .grad(&[x])
                .unwrap()
                .remove(0);
            let stablehlo = g
                .stablehlo_many(&[y.clone(), dx.clone(), h.clone()])
                .unwrap();
            assert!(!stablehlo.contains("stablehlo.divide"));
            let exe = g.compile_many(&client, &[y, dx, h]).unwrap();
            let inputs = [
                client.buffer(&shape, &data).unwrap(),
                client.buffer(&shape, &seeds).unwrap(),
                client.buffer(&shape, &direction).unwrap(),
            ];
            let outputs = exe.execute(&[&inputs[0], &inputs[1], &inputs[2]]).unwrap();
            for (output, expected) in outputs.iter().zip([prefix, gradient, hvp]) {
                assert_eq!(output.dimensions().unwrap(), shape);
                for (&actual, expected) in output.to_vec::<f32>().unwrap().iter().zip(expected) {
                    assert_eq!(actual as f64, expected, "shape={shape:?} axis={axis}");
                }
            }
        }
    }
}
