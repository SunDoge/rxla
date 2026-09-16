use rxla_pjrt::{Buffer, Client, Executable, Plugin};
use std::sync::Arc;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn persistent_pjrt_handles_are_send_and_sync() {
    assert_send_sync::<Plugin>();
    assert_send_sync::<Client>();
    assert_send_sync::<Buffer>();
    assert_send_sync::<Executable>();
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_client_and_buffer_support_concurrent_host_calls() {
    let client = unsafe {
        Client::load(std::env::var("PJRT_PLUGIN_PATH").expect("PJRT_PLUGIN_PATH must be set"))
    }
    .expect("load client");
    let shared = Arc::new(
        client
            .buffer(&[4], &[1.0_f32, 2.0, 3.0, 4.0])
            .expect("create shared buffer"),
    );

    std::thread::scope(|scope| {
        let workers = (0..4)
            .map(|worker| {
                let client = client.clone();
                let shared = Arc::clone(&shared);
                scope.spawn(move || {
                    assert_eq!(shared.dimensions().unwrap(), [4]);
                    assert_eq!(shared.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
                    let value = worker as f32;
                    assert_eq!(
                        client
                            .buffer(&[1], &[value])
                            .unwrap()
                            .to_vec::<f32>()
                            .unwrap(),
                        [value]
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}
