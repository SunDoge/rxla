use rxla_pjrt::{Client, bf16};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_host_staged_copy_preserves_scalars_empty_shapes_and_payloads() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let float = client.buffer(&[], &[-0.]).unwrap();
    let integer = client.buffer(&[2], &[i32::MIN, i32::MAX]).unwrap();
    let bf16 = client
        .buffer(&[4], &[0x8000, 0x7fc1, 0xff80, 1].map(bf16::from_bits))
        .unwrap();
    let empty = client.buffer::<f32>(&[2, 0, 3], &[]).unwrap();
    for (source, bytes) in [(float, 4), (integer, 8), (bf16, 8), (empty, 0)] {
        if bytes > 0 {
            for limit in [0, bytes - 1] {
                let error = match source.copy_to_client_via_host_with_limit(&client, limit) {
                    Ok(_) => panic!("accepted transfer above host limit"),
                    Err(error) => error,
                };
                assert!(error.to_string().contains("exceeds limit"));
            }
        }
        let copy = source
            .copy_to_client_via_host_with_limit(&client, bytes)
            .unwrap();
        assert!(copy.belongs_to(&client));
        assert_eq!(copy.dimensions().unwrap(), source.dimensions().unwrap());
        assert_eq!(copy.dtype().unwrap(), source.dtype().unwrap());
        match source.dtype().unwrap() {
            rxla_pjrt::DType::F32 => {
                assert_eq!(
                    copy.to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    source
                        .to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                )
            }
            rxla_pjrt::DType::I32 => {
                assert_eq!(
                    copy.to_vec::<i32>().unwrap(),
                    source.to_vec::<i32>().unwrap()
                )
            }
            rxla_pjrt::DType::BF16 => assert_eq!(
                copy.to_vec::<bf16>()
                    .unwrap()
                    .into_iter()
                    .map(bf16::to_bits)
                    .collect::<Vec<_>>(),
                source
                    .to_vec::<bf16>()
                    .unwrap()
                    .into_iter()
                    .map(bf16::to_bits)
                    .collect::<Vec<_>>()
            ),
            dtype => panic!("unsupported test dtype {dtype:?}"),
        }
        drop(source);
        // Destination remains usable independently of the source handle.
        assert!(copy.dtype().is_ok());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_owned_upload_works_without_backend_dma_mapping() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let upload = client
        .upload_pinned(&[4], vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap();
    assert_eq!(
        upload.wait().unwrap().to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    let empty = client.upload_pinned::<i32>(&[0, 2], vec![]).unwrap();
    assert!(empty.wait().unwrap().to_vec::<i32>().unwrap().is_empty());
}
