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

    let evicted = buffers.copy_to_memory(pinned).unwrap();
    assert!(
        evicted
            .parameters
            .iter()
            .chain(&evicted.states)
            .all(|(_, buffer)| buffer.memory_info().unwrap().kind.contains("pinned_host"))
    );
    let restored = evicted.copy_to_memory(device).unwrap();
    assert_eq!(
        restored.parameters[0].1.to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(restored.states[0].1.to_vec::<i32>().unwrap(), [7]);
}
