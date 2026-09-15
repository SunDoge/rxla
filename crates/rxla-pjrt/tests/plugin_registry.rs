use rxla_pjrt::{ClientOptions, Error, PluginRegistry};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn named_plugin_is_initialized_once_and_creates_multiple_clients() {
    let path = std::env::var("PJRT_PLUGIN_PATH").unwrap();
    let mut plugins = PluginRegistry::new();
    unsafe { plugins.register_dynamic("test", &path) }.unwrap();

    assert_eq!(plugins.names().collect::<Vec<_>>(), ["test"]);
    let first = plugins
        .create_client("test", &ClientOptions::new())
        .unwrap();
    let second = plugins
        .create_client("test", &ClientOptions::new())
        .unwrap();
    assert_eq!(
        first.info().unwrap().platform,
        second.info().unwrap().platform
    );

    let duplicate = match unsafe { plugins.register_dynamic("test", path) } {
        Ok(_) => panic!("duplicate plugin name was accepted"),
        Err(error) => error,
    };
    assert!(matches!(duplicate, Error::InvalidState { .. }));
    let missing = match plugins.plugin("missing") {
        Ok(_) => panic!("missing plugin was returned"),
        Err(error) => error,
    };
    assert!(matches!(missing, Error::InvalidArgument { .. }));
}
