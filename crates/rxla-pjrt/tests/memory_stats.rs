use rxla_pjrt::Client;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_selected_device_memory_stats() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let info = client.info().unwrap();
    let before = match client.memory_stats() {
        Ok(stats) => stats,
        Err(error) if info.platform != "cuda" => {
            eprintln!("backend memory diagnostics unavailable: {error}");
            return;
        }
        Err(error) => panic!("CUDA memory statistics failed: {error}"),
    };
    let values = vec![1.; 1024 * 1024];
    let buffer = client.buffer(&[1024 * 1024], &values).unwrap();
    assert_eq!(buffer.to_vec::<f32>().unwrap(), values);
    let after = client.memory_stats().unwrap();
    eprintln!("before={before:?}; after={after:?}");
    assert!(after.bytes_in_use >= 0);
    if info.platform == "cuda" {
        assert!(after.bytes_in_use >= before.bytes_in_use + 4 * 1024 * 1024);
    }
    if let Some(peak) = after.peak_bytes_in_use {
        assert!(peak >= after.bytes_in_use);
    }
    // Do not assume immediate reclamation after Drop or a universal allocator.
    drop(buffer);
    client.memory_stats().unwrap();
}
