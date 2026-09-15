use rxla_pjrt::{Client, DType, bf16, f16};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_bf16_transfer_preserves_all_bits_and_rejects_wrong_types() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    // Exercise every BF16 encoding, including +/-zero, subnormals, infinities,
    // quiet/signaling NaNs and their payloads. Transfer is not arithmetic.
    let bits: Vec<_> = (0..=u16::MAX).collect();
    let values = bits
        .iter()
        .copied()
        .map(bf16::from_bits)
        .collect::<Vec<_>>();
    let buffer = client.buffer(&[256, 256], &values).unwrap();
    assert_eq!(buffer.dtype().unwrap(), DType::BF16);
    assert_eq!(buffer.dimensions().unwrap(), [256, 256]);
    assert_eq!(buffer.to_vec_bf16_bits().unwrap(), bits);
    assert_eq!(
        buffer
            .to_vec::<bf16>()
            .unwrap()
            .into_iter()
            .map(bf16::to_bits)
            .collect::<Vec<_>>(),
        bits
    );
    assert!(buffer.to_vec::<f32>().is_err());
    assert!(buffer.to_vec::<i32>().is_err());
    assert!(
        client
            .buffer(&[], &[1.])
            .unwrap()
            .to_vec_bf16_bits()
            .is_err()
    );
    assert!(
        client
            .buffer(&[], &[1])
            .unwrap()
            .to_vec_bf16_bits()
            .is_err()
    );
    for shape in [vec![-1], vec![2], vec![i64::MAX, i64::MAX]] {
        assert!(client.buffer_bf16_bits(&shape, &[0x3f80]).is_err());
    }
    let empty = client.buffer_bf16_bits(&[0, 2], &[]).unwrap();
    assert_eq!(empty.dimensions().unwrap(), [0, 2]);
    assert!(empty.to_vec_bf16_bits().unwrap().is_empty());
    let scalar = client.buffer_bf16_bits(&[], &[0x3f80]).unwrap();
    assert_eq!(scalar.to_vec_bf16_bits().unwrap(), [0x3f80]);
    drop(client);
    assert_eq!(buffer.to_vec_bf16_bits().unwrap(), bits);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_f16_transfer_uses_native_half_values() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values = [
        f16::NEG_ZERO,
        f16::from_f32(1.5),
        f16::INFINITY,
        f16::from_bits(0x7e55),
    ];
    let buffer = client.buffer(&[4], &values).unwrap();
    assert_eq!(buffer.dtype().unwrap(), DType::F16);
    assert_eq!(
        buffer
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>(),
        values.into_iter().map(f16::to_bits).collect::<Vec<_>>()
    );
    assert!(buffer.to_vec::<bf16>().is_err());
}
