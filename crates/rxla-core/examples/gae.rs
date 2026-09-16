//! Explicit bootstrap/trace boundaries using ordinary tensor composition.
//! Synthetic rollout validation, not an RL trainer or environment integration.
use rxla_core::{CacheLimits, Client, Compiler, Tensor, Tracer};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn advantages(
    rewards: &Tensor,
    values: &Tensor,
    next_values: &Tensor,
    bootstrap_discounts: &Tensor,
    trace_discounts: &Tensor,
) -> std::result::Result<(Tensor, Tensor), rxla_core::Error> {
    // Caller distinguishes actual terminal states from time-limit truncation.
    // next_values must refer to the transition's next observation, not an
    // automatically reset environment's replacement observation.
    // Inputs are finite in this example; zero discounts do not sanitize NaNs.
    let delta = rewards
        .add(&bootstrap_discounts.mul(next_values)?)?
        .sub(values)?;
    let advantage = delta
        .flip(&[1])?
        .affine_scan(&trace_discounts.flip(&[1])?, 1)?
        .flip(&[1])?;
    // Rollout estimates are fixed targets, not a path through the critic graph.
    let returns = advantage.add(values)?.detach()?;
    Ok((advantage.detach()?, returns))
}

fn run() -> Result<()> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());
    let graph = Tracer::default();
    let rewards = graph.input(&[2, 5])?;
    let values = graph.input(&[2, 5])?;
    let next_values = graph.input(&[2, 5])?;
    let bootstrap = graph.input(&[2, 5])?;
    let trace = graph.input(&[2, 5])?;
    let (advantage, returns) = advantages(&rewards, &values, &next_values, &bootstrap, &trace)?;
    let detached_gradients = advantage
        .add(&returns)?
        .sum(&[0, 1], false)?
        .grad(&[values, next_values])?;
    let expressions = [
        advantage,
        returns,
        detached_gradients[0].clone(),
        detached_gradients[1].clone(),
    ];
    let gamma = 0.9_f32;
    let lambda = 0.95_f32;
    // Row 0: terminal at t=1, truncation at t=3, open rollout end at t=4.
    // Row 1: truncation at t=2, terminal at t=4. Trace cannot cross any boundary.
    let terminals = [
        false, true, false, false, false, false, false, false, false, true,
    ];
    let boundaries = [
        false, true, false, true, true, false, false, true, false, true,
    ];
    for rollout in 0..4 {
        let r: Vec<_> = (0..10)
            .map(|i| (i % 5) as f32 - 1. + rollout as f32 * 0.25)
            .collect();
        let v: Vec<_> = (0..10).map(|i| i as f32 * 0.5 - 1.).collect();
        let nv = [1_f32, 100., 2., 7., 3., -1., 2., -4., 1., 100.];
        let bd = terminals.map(|terminal| if terminal { 0. } else { gamma });
        let td = boundaries.map(|boundary| if boundary { 0. } else { gamma * lambda });
        let mut reference_advantage = [0_f64; 10];
        let mut reference_returns = [0_f64; 10];
        for row in 0..2 {
            let mut carry = 0.;
            for t in (0..5).rev() {
                let i = row * 5 + t;
                let delta = r[i] as f64 + bd[i] as f64 * nv[i] as f64 - v[i] as f64;
                carry = delta + td[i] as f64 * carry;
                reference_advantage[i] = carry;
                reference_returns[i] = carry + v[i] as f64;
            }
        }
        let exe = compiler.compile_many(&graph, &expressions)?;
        let buffers = [&r[..], &v[..], &nv[..], &bd[..], &td[..]]
            .into_iter()
            .map(|data| client.buffer(&[2, 5], data))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let outputs = exe.execute(&buffers.iter().collect::<Vec<_>>())?;
        for (output, reference) in outputs[..2]
            .iter()
            .zip([reference_advantage, reference_returns])
        {
            for (&actual, expected) in output.to_vec::<f32>()?.iter().zip(reference) {
                assert!((actual as f64 - expected).abs() < 2e-5);
            }
        }
        for gradient in &outputs[2..] {
            assert_eq!(gradient.to_vec::<f32>()?, [0.; 10]);
        }
        let target = outputs[1].to_vec::<f32>()?;
        // A terminal removes bootstrap; a truncation keeps bootstrap but stops trace.
        assert!((target[1] - r[1]).abs() < 2e-5);
        assert!((target[3] - (r[3] + gamma * nv[3])).abs() < 2e-5);
    }
    assert_eq!(compiler.stats().misses, 1);
    assert_eq!(compiler.stats().hits, 3);
    println!(
        "PASS: four two-row GAE rollouts, independent bootstrap/trace boundaries, F64 targets within 2e-5, detached critic gradients, one compilation."
    );
    Ok(())
}

fn main() -> Result<()> {
    run()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_gae_boundaries_and_detached_targets() {
    run().unwrap();
}
