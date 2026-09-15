use rxla_core::{CacheLimits, Client, Compiler, StateGraph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let mut graph = StateGraph::default();
    let cache = graph.state(&[4, 2])?;
    let value = graph.input(&[1, 2])?;
    let position_slot = graph.state_i32(&[])?;
    let position = graph.read(&position_slot)?;
    let next_position = position.wrapping_add_scalar(1)?;
    let indices = [position, graph.scalar_i32(0)?];
    let updated = graph.read(&cache)?.dynamic_update_slice(&value, &indices)?;
    graph.write_outputs(&[(&cache, updated), (&position_slot, next_position)])?;
    let selected = graph.read(&cache)?.dynamic_slice(&indices, &[1, 2])?;
    let program = graph.compile_outputs(&mut compiler, &[selected, graph.read(&position_slot)?])?;
    let mut session = program.session(vec![
        (cache.clone(), client.buffer(&[4, 2], &[0.; 8])?),
        (position_slot.clone(), client.buffer(&[], &[0])?),
    ])?;
    for position in 0..4 {
        let values = [position as f32, position as f32 + 0.5];
        let value = client.buffer(&[1, 2], &values)?;
        // Only new data: both cache and integer position are resident state.
        let result = session.run(&[&value])?;
        assert_eq!(result[0].to_vec::<f32>()?, values);
        assert_eq!(result[1].to_vec::<i32>()?, [position + 1]);
    }
    let result = session.state(&cache)?.to_vec::<f32>()?;
    assert_eq!(result, [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5]);
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(session.state(&position_slot)?.to_vec::<i32>()?, [4]);
    println!("Resident state after four calls: {result:?}");
    let saved = session.replace_state(vec![
        (cache.clone(), client.buffer(&[4, 2], &[0.; 8])?),
        (position_slot.clone(), client.buffer(&[], &[0])?),
    ])?;
    assert_eq!(session.state(&cache)?.to_vec::<f32>()?, [0.; 8]);
    assert_eq!(session.state(&position_slot)?.to_vec::<i32>()?, [0]);
    session.replace_state(saved)?;
    assert_eq!(session.state(&cache)?.to_vec::<f32>()?, result);
    assert_eq!(session.state(&position_slot)?.to_vec::<i32>()?, [4]);
    // Suspend without allocating a replacement cache. This transfers resident
    // buffers, not a host checkpoint; keep `program` to resume the session.
    let suspended = session.into_state();
    let resumed = program.session(suspended)?;
    assert_eq!(resumed.state(&cache)?.to_vec::<f32>()?, result);
    assert_eq!(resumed.state(&position_slot)?.to_vec::<i32>()?, [4]);
    println!(
        "Reset and restored cache without recompilation or state downloads during replacement"
    );
    println!("Backend compilations: {}", compiler.stats().misses);
    Ok(())
}
