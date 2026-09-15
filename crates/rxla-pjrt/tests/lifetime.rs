// Use a fresh process so other tests cannot accidentally keep the library alive.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_plugin_remains_mapped_after_last_client() {
    const CHILD: &str = "XLA_PJRT_LIFETIME_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "real_plugin_remains_mapped_after_last_client",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let path = std::fs::canonicalize(std::env::var_os("PJRT_PLUGIN_PATH").unwrap()).unwrap();
    let path = path.to_str().unwrap();
    let mapped = || {
        std::fs::read_to_string("/proc/self/maps")
            .unwrap()
            .lines()
            .any(|line| line.ends_with(path))
    };
    assert!(!mapped(), "fresh subprocess must not preload this plugin");
    {
        let client = unsafe { rxla_pjrt::Client::load(path) }.unwrap();
        let buffer = client.buffer(&[2], &[1., 2.]).unwrap();
        drop(client);
        assert_eq!(buffer.to_vec::<f32>().unwrap(), [1., 2.]);
        assert!(mapped());
    }
    assert!(
        mapped(),
        "plugin code must remain resident after the final buffer/client drop"
    );
    // Reopening the retained plugin must still support fresh client lifetimes.
    let client = unsafe { rxla_pjrt::Client::load(path) }.unwrap();
    assert_eq!(
        client.buffer(&[], &[7]).unwrap().to_vec::<i32>().unwrap(),
        [7]
    );
}
