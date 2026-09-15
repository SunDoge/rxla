use rxla_safetensors::{SafeTensors, write_buffers, write_buffers_with_metadata};
use std::io::{Cursor, Write};

#[test]
fn metadata_roundtrip_and_explicit_compatibility_without_payload_io() {
    let metadata = std::collections::HashMap::from([
        ("schema".to_owned(), "rust-tensor/train/v1".to_owned()),
        (
            "optimizer".to_owned(),
            "{\"kind\":\"adamw\",\"decay\":0.1}".to_owned(),
        ),
        ("cursor".to_owned(), "16777217".to_owned()),
        ("note".to_owned(), "训练\nresume".to_owned()),
    ]);
    let mut bytes = Vec::new();
    write_buffers_with_metadata(&mut bytes, &[], &metadata).unwrap();
    let checkpoint = SafeTensors::new(Cursor::new(bytes)).unwrap();
    assert_eq!(checkpoint.metadata(), Some(&metadata));
    checkpoint
        .require_metadata(&[("schema", "rust-tensor/train/v1"), ("cursor", "16777217")])
        .unwrap();
    for expected in [
        vec![("schema", "v2")],
        vec![("missing", "value")],
        vec![("cursor", "16777217"), ("cursor", "16777217")],
    ] {
        assert!(checkpoint.require_metadata(&expected).is_err());
    }
    assert_eq!(checkpoint.stats().reads, 0);
    assert_eq!(checkpoint.stats().uploads, 0);
    let mut bare = Vec::new();
    write_buffers(&mut bare, &[]).unwrap();
    let bare = SafeTensors::new(Cursor::new(bare)).unwrap();
    assert!(bare.metadata().is_none());
    bare.require_metadata(&[]).unwrap();
    assert!(bare.require_metadata(&[("schema", "v1")]).is_err());
}

#[test]
fn empty_export_is_valid_without_a_plugin() {
    let mut bytes = Vec::new();
    write_buffers(&mut bytes, &[]).unwrap();
    assert!(
        SafeTensors::new(Cursor::new(bytes))
            .unwrap()
            .names()
            .is_empty()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_mixed_export_roundtrip_validation_and_partial_writer_failure() {
    let client =
        unsafe { rxla_pjrt::Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let values = [f32::NEG_INFINITY, -0., 1.25, f32::INFINITY];
    let floats = client.buffer(&[2, 2], &values).unwrap();
    let integer = client.buffer(&[], &[16_777_217]).unwrap();
    let indices = client.buffer(&[3], &[i32::MIN, 0, i32::MAX]).unwrap();
    let empty = client.buffer::<f32>(&[0, 2], &[]).unwrap();
    let mut bytes = Vec::new();
    let entries = [
        ("weights", &floats),
        ("count", &integer),
        ("indices", &indices),
        ("empty", &empty),
    ];
    write_buffers(&mut bytes, &entries).unwrap();
    let mut checkpoint = SafeTensors::new(Cursor::new(bytes.clone())).unwrap();
    assert_eq!(checkpoint.read_i32("count").unwrap().values, [16_777_217]);
    assert_eq!(
        checkpoint.read_i32("indices").unwrap().values,
        [i32::MIN, 0, i32::MAX]
    );
    assert_eq!(checkpoint.read_i32("empty").unwrap().shape, [0, 2]);
    assert_eq!(
        checkpoint
            .read_f32("weights")
            .unwrap()
            .values
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        values.map(f32::to_bits)
    );
    assert_eq!(
        checkpoint
            .upload_i32(&client, "count")
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [16_777_217]
    );
    let mut reversed = Vec::new();
    write_buffers(
        &mut reversed,
        &entries.into_iter().rev().collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(bytes, reversed);
    for entries in [
        vec![("", &floats)],
        vec![("__metadata__", &floats)],
        vec![("same", &floats), ("same", &integer)],
    ] {
        let mut output = vec![42];
        assert!(write_buffers(&mut output, &entries).is_err());
        assert_eq!(output, [42]);
    }
    struct Limited {
        bytes: Vec<u8>,
        remaining: usize,
        largest: usize,
    }
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.largest = self.largest.max(bytes.len());
            let n = bytes.len().min(self.remaining);
            if n == 0 {
                return Err(std::io::Error::other("injected failure"));
            }
            self.bytes.extend_from_slice(&bytes[..n]);
            self.remaining -= n;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            panic!("must not flush")
        }
    }
    let large = client.buffer(&[40000], &vec![i32::MIN; 40000]).unwrap();
    let mut complete = Limited {
        bytes: vec![],
        remaining: usize::MAX,
        largest: 0,
    };
    write_buffers(&mut complete, &[("large", &large)]).unwrap();
    assert!(complete.largest <= 65536);
    let mut partial = Limited {
        bytes: vec![],
        remaining: 1000,
        largest: 0,
    };
    assert!(write_buffers(&mut partial, &[("large", &large)]).is_err());
    assert_eq!(partial.bytes, complete.bytes[..1000]);
    assert_eq!(large.to_vec::<i32>().unwrap(), vec![i32::MIN; 40000]);
}
