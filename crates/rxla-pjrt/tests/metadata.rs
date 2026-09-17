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
