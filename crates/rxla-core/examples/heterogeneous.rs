//! Explicit CPU -> CUDA -> CPU graphs with synchronous host staging.
//! Correctness example, not automatic placement or a performance benchmark.
use rxla_core::{CacheLimits, Client, Compiler, Graph};
use std::collections::VecDeque;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Both paths must be operator-trusted, mutually compatible native plugins.
    let cpu = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH")?) }?;
    let gpu = unsafe { Client::load(std::env::var("PJRT_CUDA_PLUGIN_PATH")?) }?;
    if !cpu.info()?.platform.eq_ignore_ascii_case("cpu")
        || !gpu.info()?.platform.eq_ignore_ascii_case("cuda")
    {
        return Err("expected distinct CPU and CUDA backends".into());
    }
    let mut cpu_compiler = Compiler::new(cpu.clone(), CacheLimits::default());
    let mut gpu_compiler = Compiler::new(gpu.clone(), CacheLimits::default());
    let preprocess = Graph::default();
    let x = preprocess.input(&[1, 2])?;
    let pre = cpu_compiler.compile(&preprocess, &x.mul_scalar(0.5)?.add_scalar(1.)?)?;
    let model = Graph::default();
    let x = model.input(&[1, 2])?;
    let w = model.input(&[2, 2])?;
    let prepared_model = model.prepare(&x.matmul(&w)?)?;
    let infer = gpu_compiler.compile_lowered(&prepared_model)?;
    let cpu_infer = cpu_compiler.compile_lowered(&prepared_model)?;
    // Identical HLO bytes do not mean shared executable/device ownership.
    assert!(!std::rc::Rc::ptr_eq(&infer, &cpu_infer));
    for (compiler, executable) in [(&mut cpu_compiler, &cpu_infer), (&mut gpu_compiler, &infer)] {
        let compiled_time = compiler.stats().compile_time;
        assert!(std::rc::Rc::ptr_eq(
            executable,
            &compiler.compile_lowered(&prepared_model)?
        ));
        assert_eq!(compiler.stats().compile_time, compiled_time);
    }
    let postprocess = Graph::default();
    let x = postprocess.input(&[1, 2])?;
    let post = cpu_compiler.compile(&postprocess, &x.sum(&[1], false)?)?;
    let weights = gpu.buffer(&[2, 2], &[2., 3., 4., 5.])?;
    let cpu_weights = cpu.buffer(&[2, 2], &[2., 3., 4., 5.])?;
    // Cross-backend transport preserves integer extremes and BF16 payload bits.
    let integers = [i32::MIN, i32::MAX, -1, 0];
    let bits = [0x8000u16, 0x7fc1, 0xff80, 0x0001];
    for source in [
        cpu.buffer(&[2, 2], &integers)?,
        cpu.buffer_bf16_bits(&[2, 2], &bits)?,
    ] {
        let remote = source.copy_to_client_via_host(&gpu)?;
        assert!(remote.belongs_to(&gpu) && !remote.belongs_to(&cpu));
        assert_eq!(remote.dtype()?, source.dtype()?);
        assert_eq!(remote.dimensions()?, [2, 2]);
        let restored = remote.copy_to_client_via_host(&cpu)?;
        match restored.dtype()? {
            rxla_core::DType::I32 => assert_eq!(restored.to_vec::<i32>()?, integers),
            rxla_core::DType::BF16 => assert_eq!(restored.to_vec_bf16_bits()?, bits),
            _ => unreachable!(),
        }
    }
    let mut serial = None;
    for capacity in [1, 3] {
        let mut staged_bytes = 0;
        let mut pending = VecDeque::new();
        let mut submitted = 0;
        let mut peak = 0;
        let mut results = [None; 17];
        while submitted < results.len() || !pending.is_empty() {
            while submitted < results.len() && pending.len() < capacity {
                let raw = [submitted as f32 - 8., submitted as f32 + 2.];
                let input = cpu.buffer(&[1, 2], &raw)?;
                let prepared = pre.submit(&[&input])?.wait()?;
                let cpu_expected =
                    cpu_infer.execute(&[&prepared[0], &cpu_weights])?[0].to_vec::<f32>()?;
                // Shape-compatible foreign buffers must not be silently accepted.
                if infer.submit(&[&prepared[0], &weights]).is_ok() {
                    return Err("GPU graph accepted a foreign CPU buffer".into());
                }
                let gpu_input = prepared[0].copy_to_client_via_host_with_limit(&gpu, 8)?;
                pending.push_back((
                    submitted,
                    infer.submit(&[&gpu_input, &weights])?,
                    cpu_expected,
                ));
                // The pending handle retains gpu_input after this scope ends.
                // Later CPU preprocessing need not explicitly wait for this GPU job.
                submitted += 1;
                peak = peak.max(pending.len());
            }
            // FIFO completion can cause head-of-line blocking. Staging remains
            // synchronous, and native submit may itself block; no overlap claim.
            let (request, execution, cpu_expected) = pending.pop_front().ok_or("empty queue")?;
            let output = execution.wait()?;
            let cpu_input = output[0].copy_to_client_via_host_with_limit(&cpu, 8)?;
            assert_eq!(cpu_input.to_vec::<f32>()?, cpu_expected);
            let result = post.submit(&[&cpu_input])?.wait()?[0].to_vec::<f32>()?;
            let a = (request as f64 - 8.) * 0.5 + 1.;
            let b = (request as f64 + 2.) * 0.5 + 1.;
            if result != [(5. * a + 9. * b) as f32] {
                return Err(format!("request {request}: incorrect result {result:?}").into());
            }
            staged_bytes += 2 * 2 * std::mem::size_of::<f32>();
            if results[request].replace(result[0].to_bits()).is_some() {
                return Err("duplicate request completion".into());
            }
        }
        assert!(results.iter().all(Option::is_some));
        assert_eq!(peak, capacity);
        if let Some(reference) = serial {
            assert_eq!(results, reference);
        } else {
            serial = Some(results);
        }
        println!(
            "CPU -> CUDA -> CPU: 17 requests verified; retained GPU tasks peak {peak}; {staged_bytes} logical inter-stage bytes via host"
        );
    }
    assert_eq!(cpu_compiler.stats().misses, 3);
    assert_eq!(gpu_compiler.stats().misses, 1);
    println!(
        "Both capacities reused 3 pipeline executables plus a CPU model check from the same prepared model; outputs match bit-for-bit for these inputs."
    );
    println!("Synchronous host staging; no zero-copy, overlap, auto-placement or speedup claim.");
    Ok(())
}
