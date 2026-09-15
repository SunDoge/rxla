use half::{bf16, f16};
use rxla_safetensors::{Dtype, SafeTensors};
use safetensors::tensor::{TensorView, serialize};
use std::io::{Cursor, Read, Seek, SeekFrom};

fn fixture(items: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    serialize(
        items.iter().map(|(name, dtype, shape, data)| {
            (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
        }),
        None,
    )
    .unwrap()
}

fn integer_fixture() -> Vec<u8> {
    fixture(&[
        (
            "indices",
            Dtype::I32,
            vec![2, 3],
            [i32::MIN, -16_777_217, -1, 0, 16_777_217, i32::MAX]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        (
            "count",
            Dtype::I32,
            vec![],
            16_777_217_i32.to_le_bytes().to_vec(),
        ),
        ("empty", Dtype::I32, vec![0, 2], vec![]),
        ("float", Dtype::F32, vec![], 1_f32.to_le_bytes().to_vec()),
        ("i64", Dtype::I64, vec![], 1_i64.to_le_bytes().to_vec()),
    ])
}

#[test]
fn quantized_u8_payload_is_read_without_conversion() {
    let bytes = fixture(&[
        (
            "weight",
            Dtype::U8,
            vec![2, 3],
            vec![0, 1, 127, 128, 254, 255],
        ),
        ("float", Dtype::F32, vec![], 1_f32.to_le_bytes().to_vec()),
    ]);
    let mut checkpoint = SafeTensors::new(Cursor::new(bytes)).unwrap();
    assert!(checkpoint.read_u8("float").is_err());
    let tensor = checkpoint.read_u8("weight").unwrap();
    assert_eq!(tensor.shape, [2, 3]);
    assert_eq!(tensor.values, [0, 1, 127, 128, 254, 255]);
    assert_eq!(checkpoint.stats().payload_bytes, 6);
}

#[test]
fn integer_payloads_preserve_bits_and_reject_implicit_casts() {
    let mut checkpoint = SafeTensors::new(Cursor::new(integer_fixture())).unwrap();
    for name in ["float", "i64", "missing"] {
        assert!(checkpoint.read_i32(name).is_err());
    }
    assert_eq!(checkpoint.stats().reads, 0);
    let values = checkpoint.read_i32("indices").unwrap();
    assert_eq!(values.shape, [2, 3]);
    assert_eq!(
        values.values,
        [i32::MIN, -16_777_217, -1, 0, 16_777_217, i32::MAX]
    );
    let count = checkpoint.read_i32("count").unwrap();
    assert!(count.shape.is_empty());
    assert_eq!(count.values, [16_777_217]);
    let empty = checkpoint.read_i32("empty").unwrap();
    assert_eq!(empty.shape, [0, 2]);
    assert!(empty.values.is_empty());
    assert_eq!(checkpoint.stats().reads, 3);
    assert_eq!(checkpoint.stats().payload_bytes, 28);
    assert_eq!(checkpoint.stats().uploads, 0);
    assert!(checkpoint.read_f32("indices").is_err());
    assert_eq!(checkpoint.stats().reads, 3);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_i32_upload_preserves_counter_and_index_values() {
    let client =
        unsafe { rxla_pjrt::Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut checkpoint = SafeTensors::new(Cursor::new(integer_fixture())).unwrap();
    for (name, shape, expected) in [
        (
            "indices",
            vec![2, 3],
            vec![i32::MIN, -16_777_217, -1, 0, 16_777_217, i32::MAX],
        ),
        ("count", vec![], vec![16_777_217]),
        ("empty", vec![0, 2], vec![]),
    ] {
        let buffer = checkpoint.upload_i32(&client, name).unwrap();
        assert_eq!(buffer.dimensions().unwrap(), shape);
        assert_eq!(buffer.to_vec::<i32>().unwrap(), expected);
    }
    assert_eq!(checkpoint.stats().reads, 3);
    assert_eq!(checkpoint.stats().uploads, 3);
    assert_eq!(checkpoint.stats().uploaded_bytes, 28);
    assert!(checkpoint.upload_i32(&client, "float").is_err());
    assert_eq!(checkpoint.stats().uploads, 3);
}

#[test]
fn float_formats_scalars_and_empty_tensors() {
    let values = [-2., 0., 1.5, f32::INFINITY, f32::NAN];
    let bytes = fixture(&[
        (
            "f32",
            Dtype::F32,
            vec![5],
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        (
            "f16",
            Dtype::F16,
            vec![5],
            values
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
        ),
        (
            "bf16",
            Dtype::BF16,
            vec![5],
            values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
        ),
        ("scalar", Dtype::F32, vec![], 42f32.to_le_bytes().to_vec()),
        ("empty", Dtype::F32, vec![0, 2], vec![]),
        ("integer", Dtype::I32, vec![1], 3i32.to_le_bytes().to_vec()),
    ]);
    let mut checkpoint = SafeTensors::new(Cursor::new(bytes)).unwrap();
    assert_eq!(checkpoint.stats().reads, 0);
    assert_eq!(checkpoint.stats().payload_bytes, 0);
    assert_eq!(checkpoint.names().len(), 6);
    for name in ["f32", "f16", "bf16"] {
        let tensor = checkpoint.read_f32(name).unwrap();
        assert_eq!(tensor.shape, [5]);
        assert_eq!(tensor.values[..4], values[..4]);
        assert!(tensor.values[4].is_nan());
    }
    assert_eq!(checkpoint.read_f32("scalar").unwrap().values, [42.]);
    assert!(checkpoint.read_f32("empty").unwrap().values.is_empty());
    assert!(checkpoint.read_f32("integer").is_err());
    assert!(checkpoint.read_f32("missing").is_err());
    assert_eq!(checkpoint.stats().reads, 5);
    assert_eq!(checkpoint.stats().payload_bytes, 44);
    assert_eq!(checkpoint.stats().uploads, 0);
    assert_eq!(checkpoint.stats().uploaded_bytes, 0);
    assert!(checkpoint.stats().upload_time.is_zero());
    let before = checkpoint.stats();
    checkpoint.read_f32("scalar").unwrap();
    assert_eq!(checkpoint.stats().reads, 6);
    assert_eq!(checkpoint.stats().payload_bytes, 48);
    assert!(checkpoint.stats().read_time >= before.read_time);
    assert!(checkpoint.stats().decode_time >= before.decode_time);
}

#[test]
fn malformed_files_are_rejected() {
    for bytes in [
        vec![],
        vec![0; 4],
        u64::MAX.to_le_bytes().to_vec(),
        [8u64.to_le_bytes().as_slice(), b"{}"].concat(),
    ] {
        assert!(SafeTensors::new(Cursor::new(bytes)).is_err());
    }
    for (header, payload) in [
        (
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            vec![0; 4],
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
            vec![0; 5],
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            vec![0; 3],
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            vec![0; 5],
        ),
    ] {
        let bytes = [
            (header.len() as u64).to_le_bytes().as_slice(),
            header.as_bytes(),
            &payload,
        ]
        .concat();
        assert!(SafeTensors::new(Cursor::new(bytes)).is_err());
    }
}

#[test]
fn only_requested_payload_is_read() {
    struct Counter {
        cursor: Cursor<Vec<u8>>,
        count: std::rc::Rc<std::cell::Cell<usize>>,
    }
    impl Read for Counter {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            let n = self.cursor.read(bytes)?;
            self.count.set(self.count.get() + n);
            Ok(n)
        }
    }
    impl Seek for Counter {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.cursor.seek(pos)
        }
    }
    let bytes = fixture(&[
        ("large", Dtype::F32, vec![1024], vec![0; 4096]),
        ("small", Dtype::F32, vec![1], 7f32.to_le_bytes().to_vec()),
    ]);
    let header_bytes = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut checkpoint = SafeTensors::new(Counter {
        cursor: Cursor::new(bytes),
        count: count.clone(),
    })
    .unwrap();
    assert_eq!(count.get(), header_bytes);
    assert_eq!(checkpoint.read_f32("small").unwrap().values, [7.]);
    assert_eq!(count.get(), header_bytes + 4);

    let bytes = fixture(&[
        ("large", Dtype::I32, vec![1024], vec![0; 4096]),
        (
            "small",
            Dtype::I32,
            vec![],
            16_777_217_i32.to_le_bytes().to_vec(),
        ),
    ]);
    let header_bytes = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    count.set(0);
    let mut checkpoint = SafeTensors::new(Counter {
        cursor: Cursor::new(bytes),
        count: count.clone(),
    })
    .unwrap();
    assert_eq!(count.get(), header_bytes);
    assert!(checkpoint.read_f32("small").is_err());
    assert_eq!(count.get(), header_bytes);
    assert_eq!(checkpoint.read_i32("small").unwrap().values, [16_777_217]);
    assert_eq!(count.get(), header_bytes + 4);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_loaded_weights_in_linear_layer() {
    use rxla_core::{Client, Graph};
    let bytes = fixture(&[
        (
            "weight",
            Dtype::BF16,
            vec![2, 3],
            [1., 2., 3., 4., 5., 6.]
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
        ),
        (
            "bias",
            Dtype::F16,
            vec![3],
            [0.5; 3]
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
        ),
    ]);
    let mut checkpoint = SafeTensors::new(Cursor::new(bytes)).unwrap();
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let weight = checkpoint.upload_f32(&client, "weight").unwrap();
    let bias = checkpoint.upload_f32(&client, "bias").unwrap();
    assert_eq!(checkpoint.stats().reads, 2);
    assert_eq!(checkpoint.stats().uploads, 2);
    assert_eq!(checkpoint.stats().payload_bytes, 18);
    assert_eq!(checkpoint.stats().uploaded_bytes, 36);
    drop(checkpoint);
    let g = Graph::default();
    let x = g.input(&[1, 2]).unwrap();
    let w = g.input(&[2, 3]).unwrap();
    let b = g.input(&[3]).unwrap();
    let y = x
        .matmul(&w)
        .unwrap()
        .add(&b.broadcast_to(&[1, 3]).unwrap())
        .unwrap();
    let executable = g.compile(&client, &y).unwrap();
    let input = client.buffer(&[1, 2], &[1., 2.]).unwrap();
    assert_eq!(
        executable.execute(&[&input, &weight, &bias]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [9.5, 12.5, 15.5]
    );
}
