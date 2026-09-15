use rxla_core::{Client, DType, Graph, InputSpec};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_tensor_artifact_cross_process() {
    use std::io::{Read, Write};
    const CHILD: &str = "XLA_TENSOR_ARTIFACT_TEST_CHILD";
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    if std::env::var_os(CHILD).is_some() {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
        // This branch never constructs a graph or calls a compile API.
        let restored =
            unsafe { rxla_core::Executable::deserialize_with_metadata(&client, &bytes) }.unwrap();
        assert_eq!(restored.input_count(), 2);
        assert_eq!(restored.output_count(), 2);
        assert_eq!(
            restored.input_spec(0),
            Some(InputSpec {
                shape: &[3],
                dtype: DType::F32
            })
        );
        assert_eq!(
            restored.input_spec(1),
            Some(InputSpec {
                shape: &[],
                dtype: DType::I32
            })
        );
        assert!(restored.input_spec(2).is_none());
        assert!(restored.input_spec(usize::MAX).is_none());
        let input = client
            .buffer(restored.input_spec(0).unwrap().shape, &[2., 4., 6.])
            .unwrap();
        let index = client
            .buffer(restored.input_spec(1).unwrap().shape, &[1])
            .unwrap();
        assert!(restored.run(&[&[2., 4., 6.]]).is_err()); // Multiple outputs.
        assert!(restored.run_many(&[&[2., 4., 6.], &[1.]]).is_err()); // I32 input.
        assert!(restored.execute(&[&input]).is_err());
        let wrong_shape = client.buffer(&[1, 3], &[2., 4., 6.]).unwrap();
        assert!(restored.execute(&[&wrong_shape, &index]).is_err());
        let wrong_dtype = client.buffer(&[], &[1.]).unwrap();
        assert!(restored.execute(&[&input, &wrong_dtype]).is_err());
        drop(client);
        let outputs = restored.execute(&[&input, &index]).unwrap();
        drop(restored);
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [5.]);
        assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [3., 5., 7.]);
        return;
    }
    let graph = Graph::default();
    let x = graph.input(&[3]).unwrap();
    let index = graph.input_i32_scalar().unwrap();
    let shifted = x.add_scalar(1.).unwrap();
    let selected = shifted.take(&index, 0).unwrap();
    let executable = graph.compile_many(&client, &[selected, shifted]).unwrap();
    assert_eq!(executable.input_count(), 2);
    assert_eq!(executable.output_count(), 2);
    assert_eq!(executable.input_spec(0).unwrap().shape, [3]);
    assert_eq!(executable.input_spec(1).unwrap().dtype, DType::I32);
    let bytes = executable.serialize_with_metadata().unwrap();
    drop(executable);
    drop(graph);
    drop(client);
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "real_tensor_artifact_cross_process",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_signature_retains_unused_inputs_and_zero_input_programs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Graph::default();
    let _unused = graph.input_i32(&[0, 3]).unwrap();
    let output = graph.constant(&[], &[7.]).unwrap();
    let executable = graph.compile(&client, &output).unwrap();
    drop(graph);
    assert_eq!(executable.input_count(), 1);
    assert_eq!(
        executable.input_spec(0),
        Some(InputSpec {
            shape: &[0, 3],
            dtype: DType::I32
        })
    );
    let input = client
        .buffer::<i32>(executable.input_spec(0).unwrap().shape, &[])
        .unwrap();
    assert_eq!(
        executable.execute(&[&input]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [7.]
    );
    let graph = Graph::default();
    let output = graph.constant(&[], &[9.]).unwrap();
    let executable = graph.compile(&client, &output).unwrap();
    assert_eq!(executable.input_count(), 0);
    assert_eq!(executable.output_count(), 1);
    assert!(executable.input_spec(0).is_none());
    assert_eq!(executable.run(&[]).unwrap(), [9.]);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_tensor_artifact_host_convenience() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Graph::default();
    let x = graph.input(&[2]).unwrap();
    let y = x.add_scalar(1.).unwrap();
    let bytes = graph
        .compile(&client, &y)
        .unwrap()
        .serialize_with_metadata()
        .unwrap();
    let restored =
        unsafe { rxla_core::Executable::deserialize_with_metadata(&client, &bytes) }.unwrap();
    assert_eq!(restored.run(&[&[2., 3.]]).unwrap(), [3., 4.]);
    assert!(restored.run(&[]).is_err());
    assert!(restored.run(&[&[1.]]).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_serialized_executable_lifetimes() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let y = x.add_scalar(1.).unwrap();
    let executable = g.compile_many(&client, &[y.clone(), y]).unwrap();
    let bytes = executable.serialize().unwrap();
    assert!(!bytes.is_empty());
    drop(executable);
    drop(g);
    for _ in 0..3 {
        let restored = unsafe { client.deserialize_executable(&bytes) }.unwrap();
        let input = client.buffer(&[2], &[2., 3.]).unwrap();
        let outputs = restored.execute(&[&input]).unwrap();
        drop(restored);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [3., 4.]);
        assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [3., 4.]);
        assert_eq!(input.to_vec::<f32>().unwrap(), [2., 3.]);
    }
}
