use rxla_core::{Client, DType, Graph};

#[test]
fn dynamic_index_validation() {
    let g = Graph::default();
    let x = g.input(&[4, 2]).unwrap();
    let i = g.input_i32_scalar().unwrap();
    let z = g.scalar_i32(0).unwrap();
    assert!(x.dynamic_slice(std::slice::from_ref(&i), &[1, 2]).is_err());
    assert!(x.dynamic_slice(&[i.clone(), z.clone()], &[5, 2]).is_err());
    assert!(x.dynamic_slice(&[i.clone(), z.clone()], &[-1, 2]).is_err());
    let other = Graph::default();
    assert!(
        x.dynamic_slice(&[other.input_i32_scalar().unwrap(), z.clone()], &[1, 2])
            .is_err()
    );
    assert!(
        x.dynamic_update_slice(&other.input(&[1, 2]).unwrap(), &[i.clone(), z.clone()])
            .is_err()
    );
    assert!(
        x.dynamic_update_slice(&g.input(&[1]).unwrap(), &[i, z])
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_runtime_indexed_cache() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Graph::default();
    let cache = g.input(&[4, 2]).unwrap();
    let update = g.input(&[1, 2]).unwrap();
    let position = g.input_i32_scalar().unwrap();
    let zero = g.scalar_i32(0).unwrap();
    let indices = [position, zero];
    let next = cache.dynamic_update_slice(&update, &indices).unwrap();
    let selected = next.dynamic_slice(&indices, &[1, 2]).unwrap();
    // Compile once; all six invocations below use the same executable.
    let exe = g.compile_many(&client, &[next, selected]).unwrap();
    assert!(exe.run_many(&[&[0.; 8], &[1., 2.], &[0.]]).is_err());
    let mut resident_cache = client.buffer(&[4, 2], &[0.; 8]).unwrap();
    let mut expected = vec![0.; 8];
    for position in [0i32, 1, 2, 3, -7, 99] {
        let values = [position as f32 + 1., position as f32 + 2.];
        let update = client.buffer(&[1, 2], &values).unwrap();
        let index = client.buffer(&[], &[position]).unwrap();
        assert_eq!(index.dtype().unwrap(), DType::I32);
        assert_eq!(index.to_vec::<i32>().unwrap(), [position]);
        assert!(index.to_vec::<f32>().is_err());
        let wrong_type = client.buffer(&[], &[position as f32]).unwrap();
        let error = exe
            .execute(&[&resident_cache, &update, &wrong_type])
            .err()
            .unwrap();
        assert!(matches!(
            error,
            rxla_core::Error::InvalidArgument { message } if message.contains("dtype")
        ));
        let mut output = exe.execute(&[&resident_cache, &update, &index]).unwrap();
        assert_eq!(output.pop().unwrap().to_vec::<f32>().unwrap(), values);
        // Updates are functional: the previous buffer remains valid and unchanged.
        assert_eq!(resident_cache.to_vec::<f32>().unwrap(), expected);
        let offset = position.clamp(0, 3) as usize * 2;
        expected[offset..offset + 2].copy_from_slice(&values);
        resident_cache = output.pop().unwrap();
        assert_eq!(resident_cache.to_vec::<f32>().unwrap(), expected);
    }
    assert!(resident_cache.to_vec::<i32>().is_err());
    assert!(client.buffer(&[2], &[1]).is_err());
    let integers = client
        .buffer(&[2, 2], &[i32::MIN, -1, 0, i32::MAX])
        .unwrap();
    assert_eq!(integers.dimensions().unwrap(), [2, 2]);
    assert_eq!(
        integers.to_vec::<i32>().unwrap(),
        [i32::MIN, -1, 0, i32::MAX]
    );
}
