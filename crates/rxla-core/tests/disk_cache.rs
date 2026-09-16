#![cfg(feature = "disk-cache")]
use rxla_core::{CacheLimits, Client, Compiler, DiskCache, Runtime, StateGraph, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn runtime_load_with_cache_restores_across_runtime_instances() {
    let plugin = std::env::var("PJRT_PLUGIN_PATH").unwrap();
    let directory = tempfile::tempdir().unwrap();
    let program = Tracer::trace(|trace| {
        let input = trace.input(&[2])?;
        Ok(vec![input.add_scalar(3.0)?])
    })
    .unwrap();

    let mut first = unsafe {
        Runtime::load_with_cache(&plugin, directory.path(), "runtime-cache-test", 1024 * 1024)
    }
    .unwrap();
    assert_eq!(
        program.run(&mut first, &[&[1.0, 2.0]]).unwrap(),
        [[4.0, 5.0]]
    );
    assert_eq!(first.stats().misses, 1);
    assert_eq!(first.stats().disk_hits, 0);
    drop(first);

    let mut restored = unsafe {
        Runtime::load_with_cache(plugin, directory.path(), "runtime-cache-test", 1024 * 1024)
    }
    .unwrap();
    assert_eq!(
        program.run(&mut restored, &[&[3.0, 4.0]]).unwrap(),
        [[6.0, 7.0]]
    );
    assert_eq!(restored.stats().misses, 0);
    assert_eq!(restored.stats().disk_hits, 1);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_lowered_program_restores_ordinary_disk_entry() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut first = compiler(&client, directory.path(), "prepared", 1024 * 1024);
    run(&mut first, 3.);
    drop(first);
    let g = Tracer::default();
    let x = g.input(&[2]).unwrap();
    let prepared = g.prepare(&x.add_scalar(3.).unwrap()).unwrap();
    let mut restored = compiler(&client, directory.path(), "prepared", 1024 * 1024);
    let exe = restored.compile_lowered(&prepared).unwrap();
    assert_eq!(exe.run(&[&[1., 2.]]).unwrap(), [4., 5.]);
    assert_eq!(restored.stats().disk_hits, 1);
    assert_eq!(restored.stats().misses, 0);
    assert!(restored.stats().compile_time.is_zero());
}

#[test]
#[ignore = "requires trusted CPU PJRT_PLUGIN_PATH"]
fn real_flags_partition_cache_across_processes() {
    const CHILD: &str = "XLA_FLAGS_CACHE_CHILD_DIRECTORY";
    const HIT: &str = "XLA_FLAGS_CACHE_EXPECT_HIT";
    if let Some(path) = std::env::var_os(CHILD) {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        assert_eq!(client.info().unwrap().platform, "cpu");
        let mut compiler = compiler(
            &client,
            std::path::Path::new(&path),
            "same-host-flags",
            1024 * 1024,
        );
        run(&mut compiler, 3.);
        let hit = std::env::var(HIT).unwrap() == "1";
        assert_eq!(compiler.stats().disk_hits, u64::from(hit));
        assert_eq!(compiler.stats().misses, u64::from(!hit));
        assert_eq!(compiler.stats().compile_time.is_zero(), hit);
        assert_eq!(compiler.stats().disk_read_errors, 0);
        assert_eq!(compiler.stats().disk_write_errors, 0);
        let inspection = unsafe { DiskCache::new(&path, "same-host-flags", 1024 * 1024) }
            .unwrap()
            .inspect()
            .unwrap();
        assert_eq!(inspection.compatible, 1);
        assert_eq!(inspection.corrupt, 0);
        assert_eq!(inspection.oversized, 0);
        assert_eq!(inspection.ignored, 0);
        assert_eq!(
            inspection.entry_files,
            inspection.compatible + inspection.incompatible
        );
        assert!(inspection.encoded_bytes > 0);
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    for (flags, hit) in [
        (None, false),
        (Some(""), false),
        (None, true),
        (Some(""), true),
    ] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "real_flags_partition_cache_across_processes",
                "--ignored",
            ])
            .env(CHILD, temp.path())
            .env(HIT, if hit { "1" } else { "0" });
        if let Some(flags) = flags {
            command.env("XLA_FLAGS", flags);
        } else {
            command.env_remove("XLA_FLAGS");
        }
        assert!(command.status().unwrap().success());
    }
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_pruned_cache_restores_with_different_original_input_numbers() {
    use std::rc::Rc;
    const CHILD: &str = "XLA_PRUNED_DISK_CACHE_TEST_PATH";
    let child_path = std::env::var_os(CHILD);
    let is_child = child_path.is_some();
    let temporary = tempfile::tempdir().unwrap();
    let path = child_path
        .as_deref()
        .map(std::path::Path::new)
        .unwrap_or(temporary.path());
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = compiler(&client, path, "pruned-same-plugin-host", 1024 * 1024);
    let mut graph = StateGraph::default();
    // Deliberately change the child source ABI. Only the compacted executable
    // is cached; original identities/mappings must come from the rebuilt graph.
    if is_child {
        graph.input(&[9]).unwrap();
    }
    let dead = graph.input(&[3]).unwrap();
    dead.relu().unwrap();
    let total = graph.state(&[]).unwrap();
    let x = graph.input(&[]).unwrap();
    let weight = graph.parameter(&[]).unwrap();
    if is_child {
        graph.input_i32(&[2]).unwrap();
    }
    let counter = graph.state_i32(&[]).unwrap();
    let next_counter = graph.input_i32_scalar().unwrap();
    let next = graph
        .read(&total)
        .unwrap()
        .add(&x.mul(weight.tensor()).unwrap())
        .unwrap();
    graph.write(&total, &next).unwrap();
    graph.write(&counter, &next_counter).unwrap();
    let program = graph
        .compile_pruned(&mut compiler, std::slice::from_ref(&next))
        .unwrap();
    let expected_indices = if is_child {
        vec![2, 3, 5]
    } else {
        vec![1, 2, 3]
    };
    assert_eq!(program.input_indices(), expected_indices);
    assert_eq!(compiler.stats().misses, u64::from(!is_child));
    assert_eq!(compiler.stats().disk_hits, u64::from(is_child));
    assert_eq!(compiler.stats().disk_read_errors, 0);
    assert_eq!(compiler.stats().disk_write_errors, 0);
    graph.compile_pruned(&mut compiler, &[next]).unwrap();
    assert_eq!(compiler.stats().hits, 1);
    let (initial, factor) = if is_child { (10., 5.) } else { (0., 2.) };
    let fresh = || {
        vec![
            (counter.clone(), client.buffer(&[], &[16_777_217]).unwrap()),
            (total.clone(), client.buffer(&[], &[initial]).unwrap()),
        ]
    };
    let shared = Rc::new(client.buffer(&[], &[factor]).unwrap());
    let mut session = program.session(fresh()).unwrap();
    session
        .bind_parameters(vec![(weight.clone(), shared.clone())])
        .unwrap();
    let mut independent = session.new_session(fresh()).unwrap();
    assert_eq!(session.input_count(), 2);
    // Rebinding by ORIGINAL numeric identity must also work after disk restore.
    session
        .bind_inputs(vec![(expected_indices[1], shared)])
        .unwrap();
    let mut expected = initial;
    for (step, value) in [3., -2., 4.].into_iter().enumerate() {
        let xb = client.buffer(&[], &[value]).unwrap();
        let counter_value = 16_777_218 + step as i32;
        let ib = client.buffer(&[], &[counter_value]).unwrap();
        expected += value * factor;
        assert_eq!(
            session.run(&[&xb, &ib]).unwrap()[0]
                .to_vec::<f32>()
                .unwrap(),
            [expected]
        );
        assert_eq!(
            session.state(&total).unwrap().to_vec::<f32>().unwrap(),
            [expected]
        );
        assert_eq!(
            session.state(&counter).unwrap().to_vec::<i32>().unwrap(),
            [counter_value]
        );
        assert_eq!(
            independent.state(&total).unwrap().to_vec::<f32>().unwrap(),
            [initial]
        );
        assert_eq!(
            independent
                .state(&counter)
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [16_777_217]
        );
    }
    let wrong = client.buffer(&[], &[1.]).unwrap();
    assert!(session.run(&[&wrong, &wrong]).is_err());
    assert_eq!(
        session.state(&total).unwrap().to_vec::<f32>().unwrap(),
        [expected]
    );
    assert_eq!(
        session.state(&counter).unwrap().to_vec::<i32>().unwrap(),
        [16_777_220]
    );
    let index = client.buffer(&[], &[99]).unwrap();
    drop((session, program, graph, compiler, client));
    assert_eq!(
        independent.run(&[&wrong, &index]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [initial + factor]
    );
    assert_eq!(
        independent
            .state(&counter)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [99]
    );
    if !is_child {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "real_pruned_cache_restores_with_different_original_input_numbers",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, path)
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(std::fs::read_dir(path).unwrap().count(), 1);
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_training_cache_restores_code_not_optimizer_state() {
    const CHILD: &str = "XLA_TRAINING_CACHE_TEST_PATH";
    let child_path = std::env::var_os(CHILD);
    let temp = tempfile::tempdir().unwrap();
    let path = child_path
        .as_deref()
        .map(std::path::Path::new)
        .unwrap_or(temp.path());
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = compiler(&client, path, "training-same-plugin-host", 8 * 1024 * 1024);
    let mut graph = StateGraph::default();
    let weight = graph.state(&[]).unwrap();
    let momentum = graph.state(&[]).unwrap();
    let steps = graph.state_i32(&[]).unwrap();
    let w = graph.read(&weight).unwrap();
    let m = graph.read(&momentum).unwrap();
    let x = graph.input(&[]).unwrap();
    let target = graph.input(&[]).unwrap();
    let rate = graph.input(&[]).unwrap();
    let error = w.mul(&x).unwrap().sub(&target).unwrap();
    let loss = error.mul(&error).unwrap();
    let grad = loss.grad(std::slice::from_ref(&w)).unwrap().remove(0);
    let next_m = m.mul_scalar(0.75).unwrap().add(&grad).unwrap();
    let next_w = w.sub(&rate.mul(&next_m).unwrap()).unwrap();
    let valid = loss
        .is_finite_mask()
        .unwrap()
        .mul(&next_m.is_finite_mask().unwrap())
        .unwrap()
        .mul(&next_w.is_finite_mask().unwrap())
        .unwrap();
    let next_step = graph.read(&steps).unwrap().wrapping_add_scalar(1).unwrap();
    graph
        .write_outputs_if(
            &valid,
            &[(&weight, next_w), (&momentum, next_m), (&steps, next_step)],
        )
        .unwrap();
    let outputs = [loss, grad, graph.read(&steps).unwrap()];
    let program = graph.compile_outputs(&mut compiler, &outputs).unwrap();
    // Rebuilding/reusing the same symbolic program also exercises memory caching.
    graph.compile_outputs(&mut compiler, &outputs).unwrap();
    assert_eq!(compiler.stats().misses, u64::from(child_path.is_none()));
    assert_eq!(compiler.stats().disk_hits, u64::from(child_path.is_some()));
    assert_eq!(compiler.stats().hits, 1);
    assert_eq!(compiler.stats().disk_read_errors, 0);
    assert_eq!(compiler.stats().disk_write_errors, 0);
    let (start_w, start_m, start_step) = if child_path.is_some() {
        (-2., 0.5, 9)
    } else {
        (1., 0., 0)
    };
    let initial = || {
        vec![
            (weight.clone(), client.buffer(&[], &[start_w]).unwrap()),
            (momentum.clone(), client.buffer(&[], &[start_m]).unwrap()),
            (steps.clone(), client.buffer(&[], &[start_step]).unwrap()),
        ]
    };
    let mut session = program.session(initial()).unwrap();
    let independent = program.session(initial()).unwrap();
    let mut expected_w = start_w as f64;
    let mut expected_m = start_m as f64;
    let mut count = start_step;
    for iteration in 0..6 {
        let x = if iteration == 2 {
            f32::NAN
        } else {
            0.5 + iteration as f32 * 0.125
        };
        let target = if child_path.is_some() { -1. } else { 3. };
        // Learning rate is runtime data and changes without recompilation.
        let rate = if iteration < 3 { 0.05f32 } else { 0.025 };
        let xb = client.buffer(&[], &[x]).unwrap();
        let yb = client.buffer(&[], &[target]).unwrap();
        let rb = client.buffer(&[], &[rate]).unwrap();
        let results = session.run(&[&xb, &yb, &rb]).unwrap();
        if x.is_nan() {
            assert!(results[0].to_vec::<f32>().unwrap()[0].is_nan());
        } else {
            let error = expected_w * x as f64 - target as f64;
            let grad = 2. * error * x as f64;
            assert!((results[0].to_vec::<f32>().unwrap()[0] as f64 - error * error).abs() < 1e-5);
            assert!((results[1].to_vec::<f32>().unwrap()[0] as f64 - grad).abs() < 1e-5);
            expected_m = 0.75 * expected_m + grad;
            expected_w -= rate as f64 * expected_m;
            count += 1;
        }
        assert!(
            (session.state(&weight).unwrap().to_vec::<f32>().unwrap()[0] as f64 - expected_w).abs()
                < 1e-5
        );
        assert!(
            (session.state(&momentum).unwrap().to_vec::<f32>().unwrap()[0] as f64 - expected_m)
                .abs()
                < 1e-5
        );
        assert_eq!(
            session.state(&steps).unwrap().to_vec::<i32>().unwrap(),
            [count]
        );
        assert_eq!(results[2].to_vec::<i32>().unwrap(), [count]);
    }
    assert_eq!(
        independent.state(&weight).unwrap().to_vec::<f32>().unwrap(),
        [start_w]
    );
    assert_eq!(
        independent
            .state(&momentum)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [start_m]
    );
    assert_eq!(
        independent.state(&steps).unwrap().to_vec::<i32>().unwrap(),
        [start_step]
    );
    assert_eq!(compiler.stats().misses, u64::from(child_path.is_none()));
    if child_path.is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "real_training_cache_restores_code_not_optimizer_state",
                "--ignored",
            ])
            .env(CHILD, path)
            .status()
            .unwrap();
        assert!(status.success());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cached_integer_position_restarts_with_fresh_state_in_child() {
    use std::rc::Rc;
    const CHILD: &str = "XLA_INTEGER_STATE_DISK_TEST_PATH";
    let child_path = std::env::var_os(CHILD);
    let temp = tempfile::tempdir().unwrap();
    let path = child_path
        .as_deref()
        .map(std::path::Path::new)
        .unwrap_or(temp.path());
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = compiler(&client, path, "integer-state-same-plugin-host", 1024 * 1024);
    let mut graph = StateGraph::default();
    let data = graph.state(&[4]).unwrap();
    let weight = graph.parameter(&[1]).unwrap();
    let position = graph.state_i32(&[]).unwrap();
    let input = graph.input(&[1]).unwrap();
    let old_position = graph.read(&position).unwrap();
    let next_data = graph
        .read(&data)
        .unwrap()
        .dynamic_update_slice(
            &input.mul(weight.tensor()).unwrap(),
            std::slice::from_ref(&old_position),
        )
        .unwrap();
    graph
        .write_outputs(&[
            (&data, next_data),
            (&position, old_position.wrapping_add_scalar(1).unwrap()),
        ])
        .unwrap();
    let program = graph.compile(&mut compiler, &[]).unwrap();
    assert_eq!(compiler.stats().misses, u64::from(child_path.is_none()));
    assert_eq!(compiler.stats().disk_hits, u64::from(child_path.is_some()));
    assert_eq!(compiler.stats().disk_read_errors, 0);
    assert_eq!(compiler.stats().disk_write_errors, 0);
    let (start, fill, factor) = if child_path.is_some() {
        (1i32, 10., 3.)
    } else {
        (0, 1., 2.)
    };
    let initial = || {
        vec![
            (position.clone(), client.buffer(&[], &[start]).unwrap()),
            (data.clone(), client.buffer(&[4], &[fill; 4]).unwrap()),
        ]
    };
    let mut first = program.session(initial()).unwrap();
    let mut second = program.session(initial()).unwrap();
    let buffer = Rc::new(client.buffer(&[1], &[factor]).unwrap());
    first
        .bind_parameters(vec![(weight.clone(), buffer.clone())])
        .unwrap();
    second.bind_parameters(vec![(weight, buffer)]).unwrap();
    assert_eq!(first.input_count(), 1);
    let mut expected = [fill; 4];
    for (step, value) in [4., 5.].into_iter().enumerate() {
        let input = client.buffer(&[1], &[value]).unwrap();
        assert!(first.run(&[&input]).unwrap().is_empty());
        expected[start as usize + step] = value * factor;
        assert_eq!(
            first.state(&data).unwrap().to_vec::<f32>().unwrap(),
            expected
        );
        assert_eq!(
            first.state(&position).unwrap().to_vec::<i32>().unwrap(),
            [start + step as i32 + 1]
        );
        assert_eq!(
            second.state(&position).unwrap().to_vec::<i32>().unwrap(),
            [start]
        );
    }
    let wrong = client.buffer(&[1], &[7]).unwrap();
    assert!(first.run(&[&wrong]).is_err());
    assert!(
        first
            .replace_state(vec![
                (data.clone(), client.buffer(&[4], &[99.; 4]).unwrap()),
                (position.clone(), client.buffer(&[], &[0.]).unwrap()),
            ])
            .is_err()
    );
    assert_eq!(
        first.state(&data).unwrap().to_vec::<f32>().unwrap(),
        expected
    );
    assert_eq!(
        first.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [start + 2]
    );
    let saved = first.replace_state(initial()).unwrap();
    first.replace_state(saved).unwrap();
    assert_eq!(
        first.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [start + 2]
    );
    let input = client.buffer(&[1], &[7.]).unwrap();
    drop((program, graph, compiler, client));
    second.run(&[&input]).unwrap();
    assert_eq!(
        second.state(&position).unwrap().to_vec::<i32>().unwrap(),
        [start + 1]
    );
    let mut second_expected = [fill; 4];
    second_expected[start as usize] = 7. * factor;
    assert_eq!(
        second.state(&data).unwrap().to_vec::<f32>().unwrap(),
        second_expected
    );
    if child_path.is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "real_cached_integer_position_restarts_with_fresh_state_in_child",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, temp.path())
            .status()
            .unwrap();
        assert!(status.success());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_cached_state_program_rebinds_fresh_sessions_across_processes() {
    use std::rc::Rc;
    const CHILD: &str = "XLA_STATE_DISK_CACHE_TEST_PATH";
    let child_path = std::env::var_os(CHILD);
    let temp = tempfile::tempdir().unwrap();
    let path = child_path
        .as_deref()
        .map(std::path::Path::new)
        .unwrap_or(temp.path());
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let mut compiler = compiler(
        &client,
        path,
        "state-test-same-plugin-and-host",
        1024 * 1024,
    );

    // Recreate the schema in each process: slot identity is intentionally local.
    // State and visible parameters are interleaved; only the native code is cached.
    let mut graph = StateGraph::default();
    let k = graph.state(&[4]).unwrap();
    let input = graph.input(&[1]).unwrap();
    let v = graph.state(&[4]).unwrap();
    let weight = graph.input(&[1]).unwrap();
    let position = graph.input_i32_scalar().unwrap();
    let update_k = input.mul(&weight).unwrap();
    let update_v = update_k.add_scalar(1.).unwrap();
    let next_k = graph
        .read(&k)
        .unwrap()
        .dynamic_update_slice(&update_k, std::slice::from_ref(&position))
        .unwrap();
    let next_v = graph
        .read(&v)
        .unwrap()
        .dynamic_update_slice(&update_v, &[position])
        .unwrap();
    graph.write_many(&[(&k, &next_k), (&v, &next_v)]).unwrap();
    let output = next_k.add(&next_v).unwrap().sum(&[0], false).unwrap();
    // Parent publishes ordinary compilation; child prepares its independently
    // reconstructed schema and must restore the same native cache entry.
    let program = if child_path.is_some() {
        let prepared = graph.prepare_outputs(&[output]).unwrap();
        prepared.compile(&mut compiler).unwrap()
    } else {
        graph.compile(&mut compiler, &[output]).unwrap()
    };
    assert_eq!(compiler.stats().misses, u64::from(child_path.is_none()));
    assert_eq!(compiler.stats().disk_hits, u64::from(child_path.is_some()));
    assert_eq!(
        compiler.stats().compile_time.is_zero(),
        child_path.is_some()
    );
    assert_eq!(compiler.stats().disk_read_errors, 0);
    assert_eq!(compiler.stats().disk_write_errors, 0);

    // Different values in the child must not be embedded in the cached executable.
    let (initial, factor) = if child_path.is_some() {
        (10., 5.)
    } else {
        (0., 2.)
    };
    let shared_weight = Rc::new(client.buffer(&[1], &[factor]).unwrap());
    let mut unrelated = StateGraph::default();
    let foreign_k = unrelated.state(&[4]).unwrap();
    let foreign_v = unrelated.state(&[4]).unwrap();
    assert!(
        program
            .session(vec![
                (foreign_k, client.buffer(&[4], &[0.; 4]).unwrap()),
                (foreign_v, client.buffer(&[4], &[0.; 4]).unwrap()),
            ])
            .is_err()
    );
    let mut first = program
        .session(vec![
            (k.clone(), client.buffer(&[4], &[initial; 4]).unwrap()),
            (v.clone(), client.buffer(&[4], &[initial + 1.; 4]).unwrap()),
        ])
        .unwrap();
    let mut second = program
        .session(vec![
            (k.clone(), client.buffer(&[4], &[100.; 4]).unwrap()),
            (v.clone(), client.buffer(&[4], &[200.; 4]).unwrap()),
        ])
        .unwrap();
    first.bind_inputs(vec![(1, shared_weight.clone())]).unwrap();
    second
        .bind_inputs(vec![(1, shared_weight.clone())])
        .unwrap();
    assert_eq!(first.input_count(), 2);
    let mut expected_k = [initial; 4];
    let mut expected_v = [initial + 1.; 4];
    for (step, index) in [0, 2, 0].into_iter().enumerate() {
        let value = step as f32 + 1.;
        let input = client.buffer(&[1], &[value]).unwrap();
        let position = client.buffer(&[], &[index]).unwrap();
        expected_k[index as usize] = value * factor;
        expected_v[index as usize] = value * factor + 1.;
        let result = first.run(&[&input, &position]).unwrap();
        assert_eq!(
            result[0].to_vec::<f32>().unwrap(),
            [expected_k.iter().sum::<f32>() + expected_v.iter().sum::<f32>()]
        );
        assert_eq!(
            first.state(&k).unwrap().to_vec::<f32>().unwrap(),
            expected_k
        );
        assert_eq!(
            first.state(&v).unwrap().to_vec::<f32>().unwrap(),
            expected_v
        );
        assert_eq!(
            second.state(&k).unwrap().to_vec::<f32>().unwrap(),
            [100.; 4]
        );
        assert_eq!(
            second.state(&v).unwrap().to_vec::<f32>().unwrap(),
            [200.; 4]
        );
    }
    // Argument failure must not commit hidden outputs on a restored executable.
    let bad = client.buffer(&[], &[1.]).unwrap();
    let index = client.buffer(&[], &[1]).unwrap();
    assert!(first.run(&[&bad, &index]).is_err());
    assert_eq!(
        first.state(&k).unwrap().to_vec::<f32>().unwrap(),
        expected_k
    );
    assert_eq!(
        first.state(&v).unwrap().to_vec::<f32>().unwrap(),
        expected_v
    );
    let input = client.buffer(&[1], &[2.]).unwrap();
    drop(shared_weight);
    drop(program);
    drop(graph);
    drop(compiler);
    drop(client);
    // Restored session owns code, fixed weights, client and independent state.
    assert_eq!(
        second.run(&[&input, &index]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [900. + 4. * factor + 1.]
    );
    if child_path.is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "real_cached_state_program_rebinds_fresh_sessions_across_processes",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, temp.path())
            .status()
            .unwrap();
        assert!(status.success());
    }
}

fn compiler(client: &Client, path: &std::path::Path, namespace: &str, limit: usize) -> Compiler {
    let disk = unsafe { DiskCache::new(path, namespace, limit) }.unwrap();
    Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(disk)
}
fn run(compiler: &mut Compiler, offset: f32) {
    let graph = Tracer::default();
    let x = graph.input(&[2]).unwrap();
    let y = x.add_scalar(offset).unwrap();
    let result = compiler
        .compile(&graph, &y)
        .unwrap()
        .run(&[&[1., 2.]])
        .unwrap();
    assert_eq!(result, [1. + offset, 2. + offset]);
}

#[test]
fn requires_explicit_namespace_and_size_limit() {
    let temp = tempfile::tempdir().unwrap();
    assert!(unsafe { DiskCache::new(temp.path(), "", 1024) }.is_err());
    assert!(unsafe { DiskCache::new(temp.path(), "test", 0) }.is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_disk_cache_cross_process() {
    const CHILD: &str = "XLA_DISK_CACHE_TEST_PATH";
    // Tests inherit the same pinned plugin and hardware/environment from parent.
    const NAMESPACE: &str = "integration-test-same-plugin-and-host";
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    if let Some(path) = std::env::var_os(CHILD) {
        let mut compiler = compiler(&client, std::path::Path::new(&path), NAMESPACE, 1024 * 1024);
        run(&mut compiler, 3.);
        assert_eq!(compiler.stats().misses, 0);
        assert_eq!(compiler.stats().disk_hits, 1);
        run(&mut compiler, 3.);
        assert_eq!(compiler.stats().hits, 1);
        compiler.clear();
        run(&mut compiler, 3.);
        assert_eq!(compiler.stats().disk_hits, 2);
        assert_eq!(compiler.stats().misses, 0);
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let mut first = compiler(&client, temp.path(), NAMESPACE, 1024 * 1024);
    run(&mut first, 3.);
    assert_eq!(first.stats().misses, 1);
    assert_eq!(first.stats().disk_hits, 0);
    assert_eq!(first.stats().disk_write_errors, 0);
    drop(first);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "real_disk_cache_cross_process",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD, temp.path())
        .status()
        .unwrap();
    assert!(status.success());
    let mut changed_namespace = compiler(
        &client,
        temp.path(),
        "different-compatibility-key",
        1024 * 1024,
    );
    run(&mut changed_namespace, 3.);
    assert_eq!(changed_namespace.stats().misses, 1);
    assert_eq!(changed_namespace.stats().disk_hits, 0);
    let mut changed_graph = compiler(&client, temp.path(), NAMESPACE, 1024 * 1024);
    run(&mut changed_graph, 4.);
    assert_eq!(changed_graph.stats().misses, 1);
    assert_eq!(changed_graph.stats().disk_hits, 0);
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_corruption_and_size_limits_fall_back_without_overwriting() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let temp = tempfile::tempdir().unwrap();
    let mut first = compiler(&client, temp.path(), "test", 1024 * 1024);
    run(&mut first, 3.);
    let entries: Vec<_> = std::fs::read_dir(temp.path()).unwrap().collect();
    assert_eq!(entries.len(), 1);
    let path = entries[0].as_ref().unwrap().path();
    // Deliberately corrupt only the test-owned file; decoder must reject it before
    // calling native deserialization, then compile normally without clobbering it.
    std::fs::write(&path, [0xff]).unwrap();
    let mut second = compiler(&client, temp.path(), "test", 1024 * 1024);
    run(&mut second, 3.);
    assert_eq!(second.stats().disk_read_errors, 1);
    assert_eq!(second.stats().misses, 1);
    assert_eq!(std::fs::read(&path).unwrap(), [0xff]);
    // Recovery is explicit and scoped to this exact lowered HLO/configuration.
    let graph = Tracer::default();
    let input = graph.input(&[2]).unwrap();
    let output = input.add_scalar(3.).unwrap();
    let program = graph.prepare(&output).unwrap();
    let other = unsafe { DiskCache::new(temp.path(), "other", 1024 * 1024) }.unwrap();
    assert!(!other.invalidate(&program).unwrap());
    assert_eq!(std::fs::read(&path).unwrap(), [0xff]);
    assert!(second.disk_cache().unwrap().invalidate(&program).unwrap());
    assert!(!second.disk_cache().unwrap().invalidate(&program).unwrap());
    // The already compiled in-memory entry still hits until explicitly cleared.
    run(&mut second, 3.);
    assert_eq!(second.stats().hits, 1);
    assert_eq!(
        second.disk_cache().unwrap().inspect().unwrap().entry_files,
        0
    );
    second.clear();
    run(&mut second, 3.);
    assert_eq!(second.stats().misses, 2);
    assert_eq!(
        second.disk_cache().unwrap().inspect().unwrap().compatible,
        1
    );
    second.clear();
    run(&mut second, 3.);
    assert_eq!(second.stats().disk_hits, 1);
    assert_eq!(second.stats().disk_read_errors, 1);
    let limited = tempfile::tempdir().unwrap();
    let mut small = compiler(&client, limited.path(), "test", 1);
    run(&mut small, 3.);
    assert_eq!(small.stats().disk_write_errors, 1);
    assert_eq!(std::fs::read_dir(limited.path()).unwrap().count(), 0);
}
