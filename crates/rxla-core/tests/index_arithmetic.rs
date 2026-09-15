use rxla_core::{Client, Graph};

#[test]
fn index_arithmetic_stays_i32_and_requires_explicit_broadcast() {
    let g = Graph::default();
    let x = g.input_i32(&[2, 3]).unwrap();
    let scalar = g.input_i32_scalar().unwrap();
    assert!(x.wrapping_add(&scalar).is_err());
    assert!(
        x.wrapping_sub(&Graph::default().input_i32(&[2, 3]).unwrap())
            .is_err()
    );
    let rhs = scalar.broadcast_to(x.shape()).unwrap();
    let value = x
        .wrapping_add(&rhs)
        .unwrap()
        .wrapping_sub(&rhs)
        .unwrap()
        .wrapping_mul(&rhs)
        .unwrap();
    let output = value.le_mask(&x).unwrap();
    let stablehlo = g.stablehlo(&output).unwrap();
    for opcode in ["add", "subtract", "multiply"] {
        assert!(stablehlo.contains(&format!("stablehlo.{opcode}")));
    }
    assert!(stablehlo.contains("tensor<2x3xi32>"));
    let empty = g.input_i32(&[0, 3]).unwrap();
    assert_eq!(empty.wrapping_add_scalar(1).unwrap().shape(), [0, 3]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_i32_arithmetic_matches_wrapping_reference_exactly() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let a = g.input_i32(&[8]).unwrap();
    let b = g.input_i32(&[8]).unwrap();
    let expected = (0..3)
        .map(|_| g.input_i32(&[8]).unwrap())
        .collect::<Vec<_>>();
    let arithmetic = [
        a.wrapping_add(&b).unwrap(),
        a.wrapping_sub(&b).unwrap(),
        a.wrapping_mul(&b).unwrap(),
    ];
    // Check signed <= in BOTH directions: all ones implies exact integer equality.
    let outputs = arithmetic
        .iter()
        .zip(&expected)
        .flat_map(|(a, b)| [a.le_mask(b).unwrap(), b.le_mask(a).unwrap()])
        .collect::<Vec<_>>();
    let exe = g.compile_many(&client, &outputs).unwrap();
    for av in [
        [i32::MAX, i32::MIN, 16_777_217, -16_777_217, 0, 1, -1, 65536],
        [0, 5, -7, i32::MIN, i32::MAX, 17, -23, -65536],
    ] {
        let bv = [1, -1, 3, -7, i32::MIN, i32::MAX, i32::MIN, 65536];
        let add: Vec<_> = av.iter().zip(bv).map(|(&a, b)| a.wrapping_add(b)).collect();
        let sub: Vec<_> = av.iter().zip(bv).map(|(&a, b)| a.wrapping_sub(b)).collect();
        let mul: Vec<_> = av.iter().zip(bv).map(|(&a, b)| a.wrapping_mul(b)).collect();
        let buffers = [&av[..], &bv[..], &add, &sub, &mul].map(|v| client.buffer(&[8], v).unwrap());
        for output in exe.execute(&buffers.iter().collect::<Vec<_>>()).unwrap() {
            assert_eq!(output.to_vec::<f32>().unwrap(), [1.; 8]);
        }
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_runtime_positions_drive_strided_gather() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let base = g.input_i32_scalar().unwrap();
    let stride = g.input_i32_scalar().unwrap();
    let offsets = g.constant_i32(&[3], &[0, 1, 2]).unwrap();
    let positions = offsets
        .wrapping_mul(&stride.broadcast_to(&[3]).unwrap())
        .unwrap()
        .wrapping_add(&base.broadcast_to(&[3]).unwrap())
        .unwrap()
        .wrapping_add_scalar(-1)
        .unwrap();
    let table = g
        .constant(&[8], &[0., 10., 20., 30., 40., 50., 60., 70.])
        .unwrap();
    let exe = g
        .compile(&client, &table.take(&positions, 0).unwrap())
        .unwrap();
    for (base, stride, expected) in [(2, 2, [10., 30., 50.]), (0, 5, [0., 40., 70.])] {
        let base = client.buffer(&[], &[base]).unwrap();
        let stride = client.buffer(&[], &[stride]).unwrap();
        assert_eq!(
            exe.execute(&[&base, &stride]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap(),
            expected
        );
    }
}
