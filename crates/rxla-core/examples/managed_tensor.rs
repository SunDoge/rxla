//! Explicit managed input -> resident input -> repeated compiled execution.
use rxla_core::{CacheLimits, Client, Compiler, DType, Storage, Tracer};
use rxla_pjrt::{ByteStrides, Shape, StridedLayout};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Owner {
    data: Vec<u8>,
    dropped: Arc<AtomicBool>,
}
impl AsRef<[u8]> for Owner {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let ints = [i32::MIN, 16_777_217, i32::MAX];
    let storage = Storage::host(
        DType::I32,
        ints.into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    let reversed = StridedLayout::new(Shape::new(&[3])?, ByteStrides::new(&[-4]), 8, 4)?;
    let integer_buffer = storage.upload(&reversed, &client)?;
    assert_eq!(integer_buffer.dtype()?, DType::I32);
    assert_eq!(
        integer_buffer.to_vec::<i32>()?,
        [i32::MAX, 16_777_217, i32::MIN]
    );
    let bits = [0x8000u16, 0x7fc1, 0x3f80];
    let storage = Storage::host(
        DType::BF16,
        bits.into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    let buffer = storage.upload(&StridedLayout::row_major(Shape::new(&[3])?, 2)?, &client)?;
    assert_eq!(buffer.dtype()?, DType::BF16);
    assert_eq!(
        buffer
            .to_vec::<rxla_core::bf16>()?
            .into_iter()
            .map(rxla_core::bf16::to_bits)
            .collect::<Vec<_>>(),
        bits
    );
    let graph = Tracer::default();
    let x = graph.input(&[3, 2])?;
    let y = x.mul(&x)?;
    let dropped = Arc::new(AtomicBool::new(false));
    let owner = Owner {
        data: (0..6).flat_map(|n| (n as f32).to_ne_bytes()).collect(),
        dropped: dropped.clone(),
    };
    let layout = StridedLayout::new(Shape::new(&[3, 2])?, ByteStrides::new(&[4, 12]), 0, 4)?;
    let host = x.with_host_storage(Storage::host(DType::F32, owner), layout)?;
    let resident = host.to_device(&client)?;
    drop(host);
    assert!(dropped.load(Ordering::Acquire)); // Upload lifetime does not leak the external owner.
    let a = resident.to_buffer(&client)?;
    let b = resident.to_buffer(&client)?;
    assert!(Arc::ptr_eq(&a, &b)); // Same native wrapper, not a second upload.
    drop((a, b));
    let mut compiler = Compiler::new(client, CacheLimits::default());
    for _ in 0..3 {
        let out = compiler.execute_bound(&y, &[&resident])?;
        assert_eq!(out[0].to_vec::<f32>()?, [0., 9., 1., 16., 4., 25.]);
    }
    let executable = compiler.compile(&graph, &y)?;
    // PendingExecution, not the Tensor descriptor, keeps this buffer alive.
    let buffer = resident.storage().unwrap().buffer().unwrap().clone();
    let pending = executable.submit(&[buffer.as_ref()])?;
    drop(buffer);
    drop(resident);
    drop(executable);
    drop(compiler);
    assert_eq!(
        pending.wait()?[0].to_vec::<f32>()?,
        [0., 9., 1., 16., 4., 25.]
    );
    println!("managed host owner, strided packing, resident reuse and in-flight lifetime: PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn managed_storage_lifetimes() {
        super::main().unwrap();
    }
}
