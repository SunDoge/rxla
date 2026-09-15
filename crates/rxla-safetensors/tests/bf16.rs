use rxla_safetensors::{Dtype, SafeTensors, save_buffers_new};
use std::io::Cursor;

#[test]
fn host_bf16_reader_is_exact_and_rejects_conversion() {
    let header = br#"{"bf":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"fp":{"dtype":"F16","shape":[1],"data_offsets":[8,10]}}"#;
    let bits = [0x8000u16, 0x0001, 0x7f81, 0xffff];
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    for value in bits {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend([0, 0]);
    let mut checkpoint = SafeTensors::new(Cursor::new(bytes)).unwrap();
    assert!(checkpoint.read_bf16_bits("fp").is_err());
    assert!(checkpoint.read_bf16_bits("missing").is_err());
    assert_eq!(checkpoint.stats().reads, 0);
    let tensor = checkpoint.read_bf16_bits("bf").unwrap();
    assert_eq!(tensor.shape, [4]);
    assert_eq!(tensor.bits, bits);
    assert_eq!(checkpoint.stats().reads, 1);
    assert_eq!(checkpoint.stats().payload_bytes, 8);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_bf16_mixed_checkpoint_round_trip() {
    let client =
        unsafe { rxla_pjrt::Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let bits: Vec<_> = (0..=u16::MAX).collect();
    let bf = client.buffer_bf16_bits(&[256, 256], &bits).unwrap();
    let empty = client.buffer_bf16_bits(&[0, 3], &[]).unwrap();
    let scalar = client.buffer_bf16_bits(&[], &[0x3f80]).unwrap();
    let fp = client.buffer(&[2], &[1., -2.]).unwrap();
    let index = client.buffer(&[], &[16_777_217]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mixed.safetensors");
    let tensors = [
        ("bf", &bf),
        ("empty", &empty),
        ("scalar", &scalar),
        ("fp", &fp),
        ("index", &index),
    ];
    save_buffers_new(&path, &tensors, None).unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(save_buffers_new(&path, &tensors, None).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let mut checkpoint = SafeTensors::open(&path).unwrap();
    let info = checkpoint.info("bf").unwrap();
    assert_eq!(info.dtype, Dtype::BF16);
    assert_eq!(info.data_offsets.1 - info.data_offsets.0, 2 * bits.len());
    let restored = checkpoint.upload_bf16_bits(&client, "bf").unwrap();
    assert_eq!(restored.to_vec_bf16_bits().unwrap(), bits);
    assert_eq!(checkpoint.stats().uploaded_bytes, 2 * bits.len() as u64);
    assert_eq!(checkpoint.stats().payload_bytes, 2 * bits.len() as u64);
    assert!(
        checkpoint
            .upload_bf16_bits(&client, "empty")
            .unwrap()
            .to_vec_bf16_bits()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        checkpoint
            .upload_bf16_bits(&client, "scalar")
            .unwrap()
            .to_vec_bf16_bits()
            .unwrap(),
        [0x3f80]
    );
    assert_eq!(checkpoint.read_f32("fp").unwrap().values, [1., -2.]);
    assert_eq!(checkpoint.read_i32("index").unwrap().values, [16_777_217]);
    let before = checkpoint.stats();
    assert!(checkpoint.upload_bf16_bits(&client, "fp").is_err());
    assert!(checkpoint.upload_bf16_bits(&client, "index").is_err());
    assert_eq!(checkpoint.stats().reads, before.reads);
    assert_eq!(checkpoint.stats().uploads, before.uploads);
}
