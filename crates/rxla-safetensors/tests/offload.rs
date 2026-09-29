#![cfg(feature = "model")]

use rxla_core::Client;
use rxla_safetensors::{Error, SchemaBuffers, WeightResidencyPool};

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
    let insufficient = WeightResidencyPool::new(device.clone(), 15);
    assert!(matches!(
        insufficient.prefetch(&evicted),
        Err(Error::ResidencyBudget {
            requested: 16,
            available: 15
        })
    ));
    assert_eq!(insufficient.reserved_bytes(), 0);

    let pool = WeightResidencyPool::new(device.clone(), 16);
    let restored = pool.prefetch(&evicted).unwrap();
    assert_eq!(pool.reserved_bytes(), 16);
    restored.block_until_ready().unwrap();
    assert_eq!(
        restored.buffers().parameters[0].1.to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(restored.buffers().states[0].1.to_vec::<i32>().unwrap(), [7]);

    // Prefetching does not consume the retained host copy.
    drop(restored);
    assert_eq!(pool.reserved_bytes(), 0);
    let restored_again = pool.prefetch(&evicted).unwrap();
    assert_eq!(
        restored_again.buffers().parameters[0]
            .1
            .to_vec::<f32>()
            .unwrap(),
        [1.0, 2.0, 3.0]
    );
}
