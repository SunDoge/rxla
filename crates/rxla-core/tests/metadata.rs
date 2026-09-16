use rxla_core::{Result, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_owned_client_metadata() {
    let client =
        unsafe { rxla_core::Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let info = client.info().unwrap();
    assert!(!info.platform.is_empty());
    assert!(!info.addressable_devices.is_empty());
    assert_eq!(
        info.addressable_devices
            .iter()
            .filter(|d| d.selected)
            .count(),
        1
    );
    assert_eq!(info, client.info().unwrap());
    let graph = Tracer::default();
    let x = graph.input(&[1]).unwrap();
    assert_eq!(
        graph.compile(&client, &x).unwrap().run(&[&[42.]]).unwrap(),
        [42.]
    );
    drop(client);
    // No borrowed native strings survive in metadata.
    assert!(!info.addressable_devices[0].kind.is_empty());
}
#[test]
fn static_metadata_and_public_result_work_without_native_execution() -> Result<()> {
    for (shape, count) in [
        (vec![], 1),
        (vec![2, 3], 6),
        (vec![2, 0, 3], 0),
        (vec![0], 0),
    ] {
        let graph = Tracer::default();
        let tensor = graph.input(&shape)?;
        let index = graph.input_i32(&shape)?;
        let outputs = [tensor.clone(), index.clone()];
        let before = graph.stablehlo_many(&outputs)?;
        assert_eq!(tensor.ndim(), shape.len());
        assert_eq!(index.ndim(), shape.len());
        assert_eq!(tensor.numel(), count);
        assert_eq!(index.numel(), count);
        assert_eq!(tensor.is_empty(), count == 0);
        assert_eq!(index.is_empty(), count == 0);
        for output in &outputs {
            assert_eq!(output.ndim(), shape.len());
            assert_eq!(output.numel(), count);
            assert_eq!(output.is_empty(), count == 0);
        }
        assert_eq!(before, graph.stablehlo_many(&outputs)?);
    }
    Ok(())
}

#[test]
fn invalid_shapes_are_rejected_before_queries_are_available() {
    let graph = Tracer::default();
    for shape in [&[-1][..], &[i64::MAX, i64::MAX][..]] {
        assert!(graph.input(shape).is_err());
        assert!(graph.input_i32(shape).is_err());
    }
}
