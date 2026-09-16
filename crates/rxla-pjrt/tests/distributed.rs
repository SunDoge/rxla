use rxla_pjrt::{
    Client, ClientOptions, DistributedClientConfig, InMemoryKeyValueStore, KeyValueStore,
};
use std::sync::Arc;

#[test]
fn in_memory_rendezvous_store_is_usable_through_the_public_trait() {
    let store: Arc<dyn KeyValueStore> = Arc::new(InMemoryKeyValueStore::default());
    store.put(b"binary\0key", b"binary\0value").unwrap();
    assert_eq!(
        store.try_get(b"binary\0key").unwrap().as_deref(),
        Some(b"binary\0value".as_slice())
    );
}

#[test]
#[ignore = "requires a trusted CUDA PJRT_PLUGIN_PATH and one visible GPU"]
fn cuda_plugin_creates_two_distributed_nodes_through_rendezvous_callbacks() {
    let path = std::env::var("PJRT_PLUGIN_PATH").unwrap();
    let store: Arc<dyn KeyValueStore> = Arc::new(InMemoryKeyValueStore::default());
    let tasks = (0..2)
        .map(|node_id| {
            let path = path.clone();
            let store = store.clone();
            std::thread::spawn(move || {
                let options = ClientOptions::new().set("visible_devices", [0_i64]);
                let config = DistributedClientConfig::new(node_id, 2, store).unwrap();
                let client = unsafe { Client::load_distributed(path, &options, config) }.unwrap();
                let info = client.info().unwrap();
                assert_eq!(info.addressable_devices.len(), 1);
                info.process_index
            })
        })
        .collect::<Vec<_>>();

    let mut process_indexes = tasks
        .into_iter()
        .map(|task| task.join().unwrap())
        .collect::<Vec<_>>();
    process_indexes.sort_unstable();
    assert_eq!(process_indexes, [0, 1]);
}
