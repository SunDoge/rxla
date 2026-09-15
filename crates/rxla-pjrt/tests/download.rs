use rxla_pjrt::Client;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn reusable_download_checks_types_lengths_and_preserves_bits() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values = [0., -0., f32::INFINITY, f32::from_bits(0x7fc01234)];
    let f = client.buffer(&[2, 2], &values).unwrap();
    let integers = [i32::MIN, i32::MAX, 16_777_217, -16_777_217];
    let i = client.buffer(&[4], &integers).unwrap();
    let bits: Vec<_> = (0..=u16::MAX).collect();
    let b = client.buffer_bf16_bits(&[256, 256], &bits).unwrap();
    let mut floats = [123.; 6];
    for len in [0, 3, 5] {
        assert!(f.copy_to(&mut floats[..len]).is_err());
        assert_eq!(floats, [123.; 6]);
    }
    assert!(i.copy_to(&mut floats[1..5]).is_err());
    assert_eq!(floats, [123.; 6]);
    for _ in 0..3 {
        f.copy_to(&mut floats[1..5]).unwrap();
        assert_eq!(floats[0], 123.);
        assert_eq!(floats[5], 123.);
        assert_eq!(
            floats[1..5].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.map(f32::to_bits)
        );
    }
    let mut ints = [77; 4];
    assert!(f.copy_to(&mut ints).is_err());
    assert_eq!(ints, [77; 4]);
    assert!(i.copy_to(&mut ints[..3]).is_err());
    assert_eq!(ints, [77; 4]);
    i.copy_to(&mut ints).unwrap();
    assert_eq!(ints, integers);
    let mut bf16 = vec![42; bits.len()];
    assert!(b.copy_to_bf16_bits(&mut bf16[..10]).is_err());
    assert!(f.copy_to_bf16_bits(&mut bf16[..4]).is_err());
    assert!(bf16.iter().all(|&v| v == 42));
    b.copy_to_bf16_bits(&mut bf16).unwrap();
    assert_eq!(bf16, bits);
    client
        .buffer::<f32>(&[0, 2], &[])
        .unwrap()
        .copy_to::<f32>(&mut [])
        .unwrap();
    client
        .buffer::<i32>(&[0], &[])
        .unwrap()
        .copy_to::<i32>(&mut [])
        .unwrap();
    client
        .buffer_bf16_bits(&[0], &[])
        .unwrap()
        .copy_to_bf16_bits(&mut [])
        .unwrap();
    let scalar = client.buffer(&[], &[19]).unwrap();
    drop(client);
    let mut output = [0];
    scalar.copy_to(&mut output).unwrap();
    assert_eq!(output, [19]);
}
