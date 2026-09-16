use rxla_core::{CacheLimits, Client, Compiler, DType, Storage, Tensor, Tracer};
use rxla_pjrt::{ByteStrides, Shape, StridedLayout};

#[test]
fn unified_dtype_validation_and_shape_operations() {
    let g = Tracer::default();
    let integer: Tensor = g.input_i32(&[2, 3]).unwrap();
    let compatibility: Tensor = integer.clone();
    assert_eq!(std::mem::size_of::<Tensor>(), std::mem::size_of::<usize>());
    for dtype in [DType::F32, DType::I32, DType::BF16] {
        let x = g.input_dtype(&[2, 3], dtype).unwrap();
        let y = x.transpose(&[1, 0]).unwrap().reshape(&[6]).unwrap();
        assert_eq!(y.dtype(), dtype);
        assert_eq!(x.take(&g.scalar_i32(0).unwrap(), 0).unwrap().dtype(), dtype);
        assert_eq!(Tensor::stack(&[x.clone(), x], 0).unwrap().dtype(), dtype);
    }
    let float = g.input(&[2, 3]).unwrap();
    assert!(float.add(&integer).is_err());
    assert!(float.take(&float, 0).is_err());
    assert!(float.take_along_axis(&float, 1).is_err());
    assert!(float.bitwise_and(&float).is_err());
    assert!(float.wrapping_add(&float).is_err());
    assert!(integer.sin().is_err());
    assert!(Tensor::concatenate(&[float.clone(), integer.clone()], 0).is_err());
    assert!(integer.select(&integer, &integer).is_err());
    assert_eq!(
        float.select(&integer, &compatibility).unwrap().dtype(),
        DType::I32
    );
    assert!(g.scalar_i32(1).unwrap().grad(&[]).is_err());
    assert!(float.sum(&[0, 1], false).unwrap().grad(&[integer]).is_err());
    assert!(float.vjp(&[], &compatibility).is_err());
    let bf16 = g.input_dtype(&[2, 3], DType::BF16).unwrap();
    assert!(bf16.mul(&bf16).is_err());
    assert_eq!(bf16.to_f32().unwrap().dtype(), DType::F32);
    assert!(
        float
            .with_host_storage(
                Storage::host(DType::I32, vec![0; 24]),
                StridedLayout::row_major(Shape::new(&[2, 3]).unwrap(), 4).unwrap(),
            )
            .is_err()
    );
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_managed_integer_and_bf16_graphs() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let g = Tracer::default();
    let ints = [i32::MIN, 16_777_217, i32::MAX];
    let x = g
        .input_dtype(&[3], DType::I32)?
        .with_host_storage(
            Storage::host(
                DType::I32,
                ints.into_iter()
                    .flat_map(i32::to_ne_bytes)
                    .collect::<Vec<_>>(),
            ),
            StridedLayout::new(Shape::new(&[3])?, ByteStrides::new(&[-4]), 8, 4)?,
        )?
        .to_device(&client)?;
    let selected = x.take(&g.constant_i32(&[3], &[2, 1, 0])?, 0)?;
    assert_eq!(
        compiler.execute_bound(&selected, &[&x])?[0].to_vec::<i32>()?,
        ints
    );
    let mask = x.lt_mask(&g.constant_i32(&[3], &[0, i32::MAX, 0])?)?;
    assert_eq!(
        compiler.execute_bound(&mask, &[&x])?[0].to_vec::<f32>()?,
        [0., 1., 1.]
    );
    let g = Tracer::default();
    let bits = [0x8000u16, 0x3f80, 0xc000];
    let b = g
        .input_dtype(&[3], DType::BF16)?
        .with_host_storage(
            Storage::host(
                DType::BF16,
                bits.into_iter()
                    .flat_map(u16::to_ne_bytes)
                    .collect::<Vec<_>>(),
            ),
            StridedLayout::row_major(Shape::new(&[3])?, 2)?,
        )?
        .to_device(&client)?;
    let reshape = b.reshape(&[1, 3])?;
    assert_eq!(
        compiler.execute_bound(&reshape, &[&b])?[0].to_vec_bf16_bits()?,
        bits
    );
    assert_eq!(
        compiler.execute_bound(&b.to_f32()?, &[&b])?[0].to_vec::<f32>()?,
        [-0., 1., -2.]
    );
    Ok(())
}
