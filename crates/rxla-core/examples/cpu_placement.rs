//! Opt-in placement preflight test using at least two logical CPU devices.
use prost::Message;
use rxla_core::{CacheLimits, Client, ClientOptions, Compiler, Graph};
use rxla_xla_proto::xla::{
    CompileOptionsProto, DeviceAssignmentProto, ExecutableBuildOptionsProto,
    device_assignment_proto::ComputationDevice,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("PJRT_PLUGIN_PATH")?;
    let device_zero_artifact = check_device(&path, 0, None)?;
    check_device(&path, 1, Some(&device_zero_artifact))?;
    let failure = match unsafe { Client::load_on_device(&path, &ClientOptions::new(), usize::MAX) }
    {
        Ok(_) => return Err("out-of-range device selection succeeded".into()),
        Err(error) => error,
    };
    assert!(failure.to_string().contains("out of range"));
    // An invalid selection must not poison subsequent native client creation.
    check_device(&path, 1, Some(&device_zero_artifact))?;
    #[cfg(feature = "disk-cache")]
    check_shared_cache(&path)?;
    Ok(())
}

#[cfg(feature = "disk-cache")]
fn check_shared_cache(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    for (round, index) in [0, 1, 0, 1, 0].into_iter().enumerate() {
        let client = unsafe { Client::load_on_device(path, &ClientOptions::new(), index) }?;
        let cache = unsafe {
            rxla_core::DiskCache::new_for_client(
                directory.path(),
                "local-trusted-cpu-build",
                &client,
                4 * 1024 * 1024,
            )?
        };
        let mut compiler =
            Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(cache);
        let graph = Graph::default();
        let x = graph.input(&[2])?;
        let executable = compiler.compile(&graph, &x.add_scalar(3.)?)?;
        let input = client.buffer(&[2], &[1., 2.])?;
        assert_eq!(executable.execute(&[&input])?[0].to_vec::<f32>()?, [4., 5.]);
        assert_eq!(compiler.stats().misses, u64::from(round < 2));
        assert_eq!(compiler.stats().disk_hits, u64::from(round >= 2));
        assert_eq!(compiler.stats().disk_read_errors, 0);
        assert_eq!(compiler.stats().disk_write_errors, 0);
        if round == 3 {
            let cache = compiler.disk_cache().unwrap();
            let inspection = cache.inspect()?;
            assert_eq!(inspection.compatible, 1);
            assert_eq!(inspection.incompatible, 1);
            assert_eq!(cache.trim_compatible_to_bytes(0)?.removed_files, 1);
            // The following round must still hit device zero's untouched file.
        }
    }
    assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
    println!(
        "Shared directory/base key: separate device artifacts and fresh-client hits; trimming device one preserves device zero's hit."
    );
    Ok(())
}

fn check_device(
    path: &str,
    index: usize,
    foreign_artifact: Option<&[u8]>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let client = unsafe { Client::load_on_device(path, &ClientOptions::new(), index) }?;
    let info = client.info()?;
    if !info.platform.eq_ignore_ascii_case("cpu") || info.addressable_devices.len() < 2 {
        return Err(
            "requires CPU plugin and XLA_FLAGS=--xla_force_host_platform_device_count=2".into(),
        );
    }
    assert!(info.addressable_devices[index].selected);
    assert_eq!(
        info.addressable_devices
            .iter()
            .filter(|d| d.selected)
            .count(),
        1
    );
    let selected = info
        .addressable_devices
        .iter()
        .find(|d| d.selected)
        .ok_or("missing selected device")?
        .id as i64;
    let other = info
        .addressable_devices
        .iter()
        .find(|d| !d.selected)
        .ok_or("missing second device")?
        .id as i64;
    let g = Graph::default();
    let input = g.input(&[2])?;
    let output = input.add_scalar(1.)?;
    let module = g.stablehlo(&output)?;
    let options = |ids: Vec<i64>| {
        CompileOptionsProto {
            executable_build_options: Some(ExecutableBuildOptionsProto {
                num_replicas: ids.len() as i64,
                num_partitions: 1,
                device_ordinal: -1,
                device_assignment: Some(DeviceAssignmentProto {
                    replica_count: ids.len() as i32,
                    computation_count: 1,
                    computation_devices: vec![ComputationDevice {
                        replica_device_ids: ids,
                    }],
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
        .encode_to_vec()
    };
    // One client owns every addressable CPU device; its selected device only
    // controls convenience uploads/default compilation.
    let other_executable = client.compile(
        rxla_pjrt::Program::mlir(module.as_bytes()),
        &options(vec![other]),
    )?;
    let replicated = client.compile(
        rxla_pjrt::Program::mlir(module.as_bytes()),
        &options(vec![selected, other]),
    )?;
    assert_eq!(replicated.device_count(), 2);
    let selected_input = client.buffer_on_device(index, &[2], &[1., 3.])?;
    let other_index = usize::from(index == 0);
    let other_input = client.buffer_on_device(other_index, &[2], &[5., 7.])?;
    assert_eq!(
        other_executable.execute(&[&other_input])?[0].to_vec::<f32>()?,
        [6., 8.]
    );
    let shards = replicated.execute_sharded(&[&[&selected_input], &[&other_input]])?;
    assert_eq!(shards[0][0].to_vec::<f32>()?, [2., 4.]);
    assert_eq!(shards[1][0].to_vec::<f32>()?, [6., 8.]);
    assert!(replicated.execute(&[&selected_input]).is_err());

    // Failed loads must not poison the surviving client or its normal path.
    let executable = client.compile(
        rxla_pjrt::Program::mlir(module.as_bytes()),
        &options(vec![selected]),
    )?;
    let input = client.buffer(&[2], &[1., 3.])?;
    assert_eq!(executable.execute(&[&input])?[0].to_vec::<f32>()?, [2., 4.]);
    let bytes = executable.serialize()?;
    let restored = unsafe { client.deserialize_executable(&bytes) }?;
    assert_eq!(restored.execute(&[&input])?[0].to_vec::<f32>()?, [2., 4.]);
    if let Some(foreign) = foreign_artifact {
        let foreign = unsafe { client.deserialize_executable(foreign) }?;
        assert_eq!(
            foreign.execute(&[&other_input])?[0].to_vec::<f32>()?,
            [6., 8.]
        );
    }
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    #[cfg(feature = "disk-cache")]
    let cache_directory = tempfile::tempdir()?;
    #[cfg(feature = "disk-cache")]
    {
        compiler = compiler.with_disk_cache(unsafe {
            rxla_core::DiskCache::new(
                cache_directory.path(),
                format!("cpu-placement-device-{selected}"),
                4 * 1024 * 1024,
            )?
        });
    }
    for _ in 0..2 {
        let executable = compiler.compile(&g, &output)?;
        assert_eq!(executable.execute(&[&input])?[0].to_vec::<f32>()?, [2., 4.]);
    }
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 1);
    #[cfg(feature = "disk-cache")]
    {
        drop(compiler);
        let mut compiler =
            Compiler::new(client.clone(), CacheLimits::default()).with_disk_cache(unsafe {
                rxla_core::DiskCache::new(
                    cache_directory.path(),
                    format!("cpu-placement-device-{selected}"),
                    4 * 1024 * 1024,
                )?
            });
        let executable = compiler.compile(&g, &output)?;
        assert_eq!(executable.execute(&[&input])?[0].to_vec::<f32>()?, [2., 4.]);
        assert_eq!(compiler.stats().misses, 0);
        assert_eq!(compiler.stats().disk_hits, 1);
    }
    println!(
        "CPU device index {index}: both addressable devices and two-replica execution passed on one client."
    );
    Ok(bytes)
}
