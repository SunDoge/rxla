use rxla_core::{
    CacheLimits, Client, Compiler, Graph,
    random::{threefry2x32, uniform_f32_from_bits},
};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_uniform_bits_have_exact_endpoints_scalar_and_empty_shapes() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    for shape in [vec![8], vec![], vec![0, 3]] {
        let graph = Graph::default();
        let bits = graph.input_i32(&shape).unwrap();
        let uniform = uniform_f32_from_bits(&bits).unwrap();
        let executable = compiler.compile(&graph, &uniform).unwrap();
        let count = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<i64>() as usize
        };
        let patterns = [
            0_u32,
            255,
            256,
            u32::MAX,
            0xffffff00,
            0x80000000,
            0x7fffffff,
            0x12345678,
        ];
        for start in [0, 3] {
            let values: Vec<_> = (0..count)
                .map(|i| patterns[(i + start) % 8] as i32)
                .collect();
            let input = client.buffer(&shape, &values).unwrap();
            let output = executable.execute(&[&input]).unwrap().remove(0);
            let expected: Vec<_> = values
                .iter()
                .map(|&v| ((v as u32 >> 8) as f64 / 16_777_216.) as f32)
                .collect();
            assert_eq!(output.dimensions().unwrap(), shape);
            assert_eq!(output.to_vec::<f32>().unwrap(), expected);
            assert!(expected.iter().all(|&x| (0. ..1.).contains(&x)));
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_graph_uniform_drives_dropout_without_host_masks() {
    const N: usize = 8192;
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let graph = Graph::default();
    let key0 = graph
        .input_i32_scalar()
        .unwrap()
        .broadcast_to(&[N as i64])
        .unwrap();
    let key1 = graph
        .input_i32_scalar()
        .unwrap()
        .broadcast_to(&[N as i64])
        .unwrap();
    let base = graph
        .input_i32_scalar()
        .unwrap()
        .broadcast_to(&[N as i64])
        .unwrap();
    let x = graph.input(&[N as i64]).unwrap();
    let low = graph
        .iota_i32(&[N as i64], 0)
        .unwrap()
        .wrapping_add(&base)
        .unwrap();
    let high = graph.constant_i32(&[N as i64], &vec![0; N]).unwrap();
    let bits = threefry2x32([&key0, &key1], [&low, &high]).unwrap();
    let uniform = uniform_f32_from_bits(&bits[0]).unwrap();
    let half = graph
        .constant(&[], &[0.5])
        .unwrap()
        .broadcast_to(&[N as i64])
        .unwrap();
    let mask = uniform.lt_mask(&half).unwrap();
    let dropped = x.dropout_with_mask(&mask, 0.5).unwrap();
    let gradient = dropped
        .sum(&[0], false)
        .unwrap()
        .grad(std::slice::from_ref(&x))
        .unwrap()
        .remove(0);
    let executable = compiler
        .compile_outputs(&graph, &[bits[0].clone(), uniform, mask, dropped, gradient])
        .unwrap();
    let values: Vec<_> = (0..N).map(|i| (i % 17) as f32 - 8.).collect();
    let input = client.buffer(&[N as i64], &values).unwrap();
    let k0 = client.buffer(&[], &[0x12345678]).unwrap();
    let k1 = client.buffer(&[], &[0xdeadbeef_u32 as i32]).unwrap();
    let mut previous = None;
    for offset in [0, N as i32] {
        let base = client.buffer(&[], &[offset]).unwrap();
        let out = executable.execute(&[&k0, &k1, &base, &input]).unwrap();
        let raw = out[0].to_vec::<i32>().unwrap();
        let u = out[1].to_vec::<f32>().unwrap();
        let mask = out[2].to_vec::<f32>().unwrap();
        let actual = out[3].to_vec::<f32>().unwrap();
        let grad = out[4].to_vec::<f32>().unwrap();
        for i in 0..N {
            let expected = ((raw[i] as u32 >> 8) as f64 / 16_777_216.) as f32;
            let keep = if expected < 0.5 { 1. } else { 0. };
            assert_eq!(u[i], expected);
            assert_eq!(mask[i], keep);
            assert_eq!(actual[i], values[i] * keep * 2.);
            assert_eq!(grad[i], keep * 2.);
            assert!((0. ..1.).contains(&u[i]));
        }
        let mean = u.iter().map(|&v| f64::from(v)).sum::<f64>() / N as f64;
        assert!((mean - 0.5).abs() < 0.02);
        assert!((mask.iter().sum::<f32>() / N as f32 - 0.5).abs() < 0.03);
        if let Some(previous) = previous {
            assert_ne!(raw, previous);
        }
        previous = Some(raw);
    }
    assert_eq!(compiler.stats().misses, 1);
}
