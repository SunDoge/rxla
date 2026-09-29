use rxla_pjrt::Client;

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_platform_and_device_metadata_are_owned() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let info = client.info().unwrap();
    drop(client);
    assert!(!info.platform.is_empty());
    assert!(
        info.addressable_devices
            .iter()
            .any(|d| d.selected && !d.kind.is_empty())
    );
    println!("PJRT metadata: {info:?}");
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn buffer_reports_device_and_backend_memory_independently() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let buffer = client.buffer(&[2], &[1.0_f32, 2.0]).unwrap();

    let device = buffer.device_info().unwrap();
    let memory = buffer.memory_info().unwrap();

    assert!(device.selected);
    assert!(!device.kind.is_empty());
    assert!(client.info().unwrap().addressable_devices.contains(&device));
    println!("PJRT buffer placement: device={device:?}, memory={memory:?}");
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH with memory-space copies"]
fn buffer_round_trips_through_pinned_host_memory() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let memories = client.memories().unwrap();
    let infos = memories
        .iter()
        .map(|memory| memory.info().unwrap())
        .collect::<Vec<_>>();
    println!("PJRT addressable memories: {infos:?}");

    let source = client.buffer(&[4], &[1.0_f32, -2.0, 3.5, 0.0]).unwrap();
    let device_id = source.memory_info().unwrap().id;
    let pinned = memories
        .iter()
        .find(|memory| memory.info().unwrap().kind.contains("pinned_host"))
        .expect("CUDA PJRT exposes pinned_host memory");
    let device = memories
        .iter()
        .find(|memory| memory.info().unwrap().id == device_id)
        .expect("buffer memory is addressable");

    let evicted = source.copy_to_memory(pinned).unwrap();
    assert!(evicted.memory_info().unwrap().kind.contains("pinned_host"));
    let restored = evicted.copy_to_memory(device).unwrap();
    assert_eq!(restored.memory_info().unwrap().id, device_id);
    assert_eq!(restored.to_vec::<f32>().unwrap(), [1.0, -2.0, 3.5, 0.0]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH with pinned-host memory"]
fn host_tensor_uploads_directly_to_pinned_memory() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let pinned = client
        .memories()
        .unwrap()
        .into_iter()
        .find(|memory| memory.info().unwrap().kind.contains("pinned_host"))
        .expect("CUDA PJRT exposes pinned_host memory");
    let buffer = client
        .buffer_in_memory(&pinned, &[3], &[1.0_f32, 2.0, 3.0])
        .unwrap();

    assert!(buffer.memory_info().unwrap().kind.contains("pinned_host"));
    assert_eq!(buffer.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
}
