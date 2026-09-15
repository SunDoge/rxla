use rxla_safetensors::{SafeTensors, save_buffers_new};
use std::{
    collections::HashMap,
    sync::{Arc, Barrier},
};

#[test]
fn competing_saves_publish_one_complete_file_without_overwriting() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("step-1.safetensors");
    let barrier = Arc::new(Barrier::new(2));
    let tasks: Vec<_> = (0..2)
        .map(|i| {
            let barrier = barrier.clone();
            let target = target.clone();
            std::thread::spawn(move || {
                let metadata = HashMap::from([("writer".to_owned(), i.to_string())]);
                barrier.wait();
                save_buffers_new(target, &[], Some(&metadata)).is_ok()
            })
        })
        .collect();
    let results: Vec<_> = tasks.into_iter().map(|task| task.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|&&ok| ok).count(), 1);
    let checkpoint = SafeTensors::open(&target).unwrap();
    let winner = results.iter().position(|ok| *ok).unwrap().to_string();
    checkpoint.require_metadata(&[("writer", &winner)]).unwrap();
    let before = std::fs::read(&target).unwrap();
    assert!(save_buffers_new(&target, &[], None).is_err());
    assert_eq!(std::fs::read(&target).unwrap(), before);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn missing_parent_is_not_created() {
    let directory = tempfile::tempdir().unwrap();
    assert!(save_buffers_new(directory.path().join("missing/checkpoint"), &[], None).is_err());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn existing_dangling_symlink_is_not_followed_or_replaced() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("checkpoint");
    let missing = directory.path().join("missing");
    std::os::unix::fs::symlink(&missing, &target).unwrap();
    assert!(save_buffers_new(&target, &[], None).is_err());
    assert_eq!(std::fs::read_link(&target).unwrap(), missing);
    assert!(!missing.exists());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_save_validation_failure_cleans_temp_and_success_preserves_payloads() {
    let client =
        unsafe { rxla_pjrt::Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let value = client.buffer(&[], &[16_777_217]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("step.safetensors");
    assert!(save_buffers_new(&target, &[("same", &value), ("same", &value)], None).is_err());
    assert!(!target.exists());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    save_buffers_new(&target, &[("step", &value)], None).unwrap();
    assert_eq!(
        SafeTensors::open(&target)
            .unwrap()
            .read_i32("step")
            .unwrap()
            .values,
        [16_777_217]
    );
    assert_eq!(value.to_vec::<i32>().unwrap(), [16_777_217]);
}
