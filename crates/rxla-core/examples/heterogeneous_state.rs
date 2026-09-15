//! Explicit quiescent session snapshots between CPU and CUDA, via host memory.
//! One prepared transition compiled per backend; not live migration or executable reuse.
use rxla_core::{CacheLimits, Client, Compiler, F32, I32, Session, State, StateGraph};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn check(session: &Session, value: &State<F32>, count: &State<I32>, steps: i32) -> Result<()> {
    assert_eq!(
        session.state(value.as_slot())?.to_vec::<f32>()?,
        [1. + 2. * steps as f32, 3. + 2. * steps as f32]
    );
    assert_eq!(
        session.state(count.as_slot())?.to_vec::<i32>()?,
        [i32::MAX.wrapping_add(steps)]
    );
    Ok(())
}

fn main() -> Result<()> {
    let cpu = unsafe { Client::load(std::env::var("PJRT_CPU_PLUGIN_PATH")?) }?;
    let gpu = unsafe { Client::load(std::env::var("PJRT_CUDA_PLUGIN_PATH")?) }?;
    if !cpu.info()?.platform.eq_ignore_ascii_case("cpu")
        || !gpu.info()?.platform.eq_ignore_ascii_case("cuda")
    {
        return Err("expected CPU and CUDA backends".into());
    }
    let (prepared, value, count) = {
        let mut g = StateGraph::default();
        g.input(&[3])?; // Unused input must not become a runtime requirement.
        let value = State::<F32>::new(&mut g, &[2])?;
        let count = State::<I32>::new(&mut g, &[])?;
        let accept = g.input(&[])?;
        let next_value = value.read(&g)?.add_scalar(2.)?;
        let next_count = count.read(&g)?.wrapping_add_scalar(1)?;
        g.transaction()
            .with(&value, &next_value)?
            .with(&count, &next_count)?
            .commit_if(&accept)?;
        (g.prepare_outputs_pruned(&[value.read(&g)?])?, value, count)
    }; // Source graph and all symbolic tensor handles are gone before compilation.
    let mut cpu_compiler = Compiler::new(cpu.clone(), CacheLimits::default());
    let mut gpu_compiler = Compiler::new(gpu.clone(), CacheLimits::default());
    let cpu_program = prepared.compile(&mut cpu_compiler)?;
    let gpu_program = prepared.compile(&mut gpu_compiler)?;
    assert_eq!(cpu_program.input_indices(), [1]);
    assert_eq!(gpu_program.input_indices(), [1]);
    for compiler in [&mut cpu_compiler, &mut gpu_compiler] {
        let time = compiler.stats().compile_time;
        prepared.compile(compiler)?;
        assert_eq!(compiler.stats().hits, 1);
        assert_eq!(compiler.stats().compile_time, time);
    }
    drop(prepared); // Compiled programs own their native code and schema.
    let mut baseline = cpu_program.session(vec![
        (value.as_slot().clone(), cpu.buffer(&[2], &[1., 3.])?),
        (count.as_slot().clone(), cpu.buffer(&[], &[i32::MAX])?),
    ])?;
    let yes = cpu.buffer(&[], &[1.])?;
    let no = cpu.buffer(&[], &[0.])?;
    baseline.run(&[&yes])?;
    check(&baseline, &value, &count, 1)?;
    assert!(baseline.copy_state_to_client_via_host(&gpu, 7).is_err());
    // Each tensor fits eight bytes, but all slots together require twelve.
    assert!(
        baseline
            .copy_state_to_client_via_host_with_limits(&gpu, 8, 11)
            .is_err()
    );
    check(&baseline, &value, &count, 1)?;
    let mut remote =
        gpu_program.session(baseline.copy_state_to_client_via_host_with_limits(&gpu, 8, 12)?)?;
    check(&remote, &value, &count, 1)?;
    let gpu_yes = yes.copy_to_client_via_host(&gpu)?;
    let gpu_no = no.copy_to_client_via_host(&gpu)?;
    let mut steps = 1;
    for accept in [false, true, true, false] {
        let output = remote.run(&[if accept { &gpu_yes } else { &gpu_no }])?;
        if accept {
            steps += 1;
        }
        check(&remote, &value, &count, steps)?;
        // Copying creates independent state: the source is not advanced by remote work.
        check(&baseline, &value, &count, 1)?;
        assert_eq!(
            output[0].to_vec::<f32>()?,
            [1. + 2. * steps as f32, 3. + 2. * steps as f32]
        );
    }
    for mask in [&no, &yes, &yes, &no] {
        baseline.run(&[mask])?;
    }
    check(&baseline, &value, &count, steps)?;
    assert!(
        remote
            .copy_state_to_client_via_host_with_limits(&cpu, 8, 11)
            .is_err()
    );
    check(&remote, &value, &count, steps)?;
    remote.run(&[&gpu_no])?;
    check(&remote, &value, &count, steps)?;
    let mut returned =
        cpu_program.session(remote.copy_state_to_client_via_host_with_limits(&cpu, 8, 12)?)?;
    assert_eq!(gpu_compiler.stats().misses, 1);
    // Destination remains usable after destroying the source session/backend.
    drop((remote, gpu_yes, gpu_no, gpu_program, gpu_compiler, gpu));
    for mask in [&yes, &no, &yes] {
        let expected = baseline.run(&[mask])?;
        let actual = returned.run(&[mask])?;
        assert_eq!(actual[0].to_vec::<f32>()?, expected[0].to_vec::<f32>()?);
        assert_eq!(
            returned.state(count.as_slot())?.to_vec::<i32>()?,
            baseline.state(count.as_slot())?.to_vec::<i32>()?
        );
    }
    check(&returned, &value, &count, steps + 2)?;
    assert_eq!(cpu_compiler.stats().misses, 1);
    println!(
        "CPU -> CUDA -> CPU state continuation passed: one prepared pruned transition, F32 values, I32 wraparound, rejected steps, source isolation, total-payload budget rejection and exact-boundary copies; one compilation per backend."
    );
    println!(
        "Synchronous copied snapshots of a quiescent session, not live migration or cross-backend executable portability."
    );
    Ok(())
}
