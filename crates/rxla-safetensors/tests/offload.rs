#![cfg(feature = "model")]

use rxla_core::Client;
use rxla_safetensors::SchemaBuffers;

#[test]
#[ignore = "requires trusted PJRT_CUDA_PLUGIN_PATH"]
fn schema_buffers_round_trip_through_pinned_host_memory() {
    let client = unsafe { Client::load(std::env::var("PJRT_CUDA_PLUGIN_PATH").unwrap()) }.unwrap();
    let memories = client.memories().unwrap();
    let pinned = memories
        .iter()
        .find(|memory| memory.info().unwrap().kind.contains("pinned_host"))
        .unwrap();
    let device = memories
        .iter()
        .find(|memory| memory.info().unwrap().kind == "device")
        .unwrap();
    let buffers = SchemaBuffers {
        parameters: vec![(
            "weight".to_owned(),
            client.buffer(&[3], &[1.0_f32, 2.0, 3.0]).unwrap(),
        )],
        states: vec![("counter".to_owned(), client.buffer(&[1], &[7_i32]).unwrap())],
    };

    let evicted = buffers.offload_to(pinned).unwrap();
    assert_eq!(evicted.payload_bytes(), 16);
    assert!(
        evicted
            .memory_info()
            .unwrap()
            .unwrap()
            .kind
            .contains("pinned_host")
    );
    let restored = evicted.prefetch_to(device).unwrap();
    restored.block_until_ready().unwrap();
    assert_eq!(
        restored.buffers().parameters[0].1.to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(restored.buffers().states[0].1.to_vec::<i32>().unwrap(), [7]);

    // Prefetching does not consume the retained host copy.
    let restored_again = evicted.prefetch_to(device).unwrap();
    assert_eq!(
        restored_again.buffers().parameters[0]
            .1
            .to_vec::<f32>()
            .unwrap(),
        [1.0, 2.0, 3.0]
    );
}
