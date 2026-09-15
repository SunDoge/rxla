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
