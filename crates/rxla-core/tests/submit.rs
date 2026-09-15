use rxla_core::{Client, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_device_pipeline_with_shared_intermediate_outputs() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let i = g.input_i32(&[2]).unwrap();
    let exe = g
        .compile_outputs(
            &client,
            &[x.add_scalar(1.).unwrap(), i.wrapping_add_scalar(1).unwrap()],
        )
        .unwrap();
    let input = client.buffer(&[2], &[3., -7.]).unwrap();
    let words = client.buffer(&[2], &[i32::MAX, 16_777_217]).unwrap();
    let first = exe.submit(&[&input, &words]).unwrap();
    // Fan out and chain native buffers without a host wait or download between
    // stages. Each submit owns its input references; no donation is permitted.
    let second = exe
        .submit(&[&first.outputs()[0], &first.outputs()[1]])
        .unwrap();
    let branch = exe
        .submit(&[&first.outputs()[0], &first.outputs()[1]])
        .unwrap();
    let third = exe
        .submit(&[&second.outputs()[0], &second.outputs()[1]])
        .unwrap();
    drop(input);
    drop(words);
    drop(exe);
    drop(client);
    drop(g);
    // Wait downstream first. Checking the upstream tasks separately retains
    // their error reporting, even though their outputs have downstream users.
    let third = third.wait().unwrap();
    let first = first.wait().unwrap();
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [4., -6.]);
    assert_eq!(first[1].to_vec::<i32>().unwrap(), [i32::MIN, 16_777_218]);
    drop(first);
    for stage in [second, branch] {
        let outputs = stage.wait().unwrap();
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [5., -5.]);
        assert_eq!(
            outputs[1].to_vec::<i32>().unwrap(),
            [i32::MIN + 1, 16_777_219]
        );
    }
    assert_eq!(third[0].to_vec::<f32>().unwrap(), [6., -4.]);
    assert_eq!(
        third[1].to_vec::<i32>().unwrap(),
        [i32::MIN + 2, 16_777_220]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_completion_queries_preserve_outputs_and_owners() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let exe = g.compile(&client, &x.add_scalar(2.).unwrap()).unwrap();
    let input = client.buffer(&[2], &[3., -7.]).unwrap();
    let pending = exe.submit(&[&input]).unwrap();
    drop(input);
    drop(exe);
    drop(client);
    drop(g);
    // Fast backends may already be ready at the first query. Never require an
    // initial false result or infer device overlap from observed readiness.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !pending.is_ready().unwrap() {
        assert!(std::time::Instant::now() < deadline, "completion timeout");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    for _ in 0..3 {
        assert!(pending.is_ready().unwrap());
    }
    assert_eq!(
        pending.wait().unwrap()[0].to_vec::<f32>().unwrap(),
        [5., -5.]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_submit_without_inputs_retains_constant_executable() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let value = g.constant(&[2], &[3., -7.]).unwrap();
    let exe = g.compile(&client, &value).unwrap();
    let pending = exe.submit(&[]).unwrap();
    drop(exe);
    drop(client);
    drop(g);
    assert_eq!(
        pending.wait().unwrap()[0].to_vec::<f32>().unwrap(),
        [3., -7.]
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_submissions_own_inputs_and_executable_until_wait() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2, 2]).unwrap();
    let i = g.input_i32(&[2]).unwrap();
    let exe = g
        .compile_outputs(
            &client,
            &[x.matmul(&x).unwrap(), i.wrapping_add_scalar(1).unwrap()],
        )
        .unwrap();
    let a = client.buffer(&[2, 2], &[1., 2., 3., 4.]).unwrap();
    let b = client.buffer(&[2, 2], &[2., 0., 0., 3.]).unwrap();
    let integers = client.buffer(&[2], &[i32::MAX, 16_777_217]).unwrap();
    let first = exe.submit(&[&a, &integers]).unwrap();
    let second = exe.submit(&[&b, &integers]).unwrap();
    drop(a);
    drop(b);
    drop(integers);
    drop(exe);
    drop(client);
    drop(g);
    // Reverse wait order is legal; this does not demonstrate parallel kernels.
    let second = second.wait().unwrap();
    let first = first.wait().unwrap();
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [7., 10., 15., 22.]);
    assert_eq!(second[0].to_vec::<f32>().unwrap(), [4., 0., 0., 9.]);
    for outputs in [first, second] {
        assert_eq!(outputs[1].to_vec::<i32>().unwrap(), [i32::MIN, 16_777_218]);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_submit_validates_signatures_and_drop_does_not_cancel() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let x = g.input(&[2]).unwrap();
    let exe = g.compile(&client, &x.add_scalar(1.).unwrap()).unwrap();
    let good = client.buffer(&[2], &[1., 2.]).unwrap();
    let wrong_shape = client.buffer(&[], &[1.]).unwrap();
    let wrong_dtype = client.buffer(&[2], &[1, 2]).unwrap();
    assert!(exe.submit(&[]).is_err());
    assert!(exe.submit(&[&wrong_shape]).is_err());
    assert!(exe.submit(&[&wrong_dtype]).is_err());
    for _ in 0..4 {
        drop(exe.submit(&[&good]).unwrap());
    }
    assert_eq!(
        exe.execute(&[&good]).unwrap()[0].to_vec::<f32>().unwrap(),
        [2., 3.]
    );
    assert_eq!(good.to_vec::<f32>().unwrap(), [1., 2.]);
    let pending = exe.submit(&[&good]).unwrap();
    drop(good);
    drop(exe);
    drop(client);
    drop(pending); // must keep owners until completion even without explicit wait
}
