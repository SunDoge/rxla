#![cfg(feature = "model")]

use rxla_core::{Buffer, CacheLimits, Client, Compiler, Tensor};
use rxla_nn::{Cx, Model, ModelInput};
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

    let pressured =
        WeightResidencyPool::new(device.clone(), 16).with_allocator_headroom(usize::MAX);
    assert!(matches!(
        pressured.prefetch(&evicted),
        Err(Error::DeviceMemoryPressure { .. })
    ));
    assert_eq!(pressured.reserved_bytes(), 0);

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
    let model = Model::new(|cx: Cx, input: Tensor| -> rxla_nn::Result<Tensor> {
        let weight = cx.param("weight", [3])?;
        Ok(input.add(&weight)?)
    })
    .inputs(ModelInput::new([3]))
    .trace()
    .unwrap();
    let compiled = model
        .compile(&mut Compiler::new(client.clone(), CacheLimits::default()))
        .unwrap();
    let runner = restored.bind(&compiled).unwrap();
    let input = client.buffer(&[3], &[10.0_f32, 20.0, 30.0]).unwrap();
    let output: Buffer = runner.run(&input).unwrap();
    assert_eq!(output.to_vec::<f32>().unwrap(), [11.0, 22.0, 33.0]);
    assert_eq!(pool.reserved_bytes(), 16);
    drop(runner);
    assert_eq!(pool.reserved_bytes(), 0);
    let pending = pool.prefetch_async(evicted.clone());
    let restored_again = pending.wait().unwrap();
    assert_eq!(
        restored_again.buffers().parameters[0]
            .1
            .to_vec::<f32>()
            .unwrap(),
        [1.0, 2.0, 3.0]
    );
}
