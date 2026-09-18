//! A small end-to-end policy-gradient experiment on CartPole.
//!
//! Environment simulation and trajectory bookkeeping stay on the host. Policy
//! inference, the REINFORCE objective, autodiff, and Adam execute through RXLA.

use clap::Parser;
use rxla_core::{Buffer, CacheLimits, Client, Compiler, DType, Tensor};
use rxla_nn::{Cx, Linear, Model, ModelInput, Result as NnResult, TensorApply};
use rxla_train::{AdamOptions, prepare_model_adam};
use std::{collections::BTreeMap, f32::consts::PI};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "PJRT_PLUGIN_PATH")]
    plugin: String,
    #[arg(long, default_value_t = 150)]
    iterations: usize,
    #[arg(long, default_value_t = 32)]
    environments: usize,
    #[arg(long, default_value_t = 32)]
    horizon: usize,
    #[arg(long, default_value_t = 0.99)]
    gamma: f32,
    #[arg(long, default_value_t = 3e-3)]
    learning_rate: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Clone, Copy)]
struct CartPole {
    state: [f32; 4],
    episode_return: f32,
}

impl CartPole {
    fn reset(rng: &mut Rng) -> Self {
        let mut state = [0.; 4];
        for value in &mut state {
            *value = (rng.uniform() - 0.5) * 0.1;
        }
        Self {
            state,
            episode_return: 0.,
        }
    }

    fn step(&mut self, action: usize) -> bool {
        const GRAVITY: f32 = 9.8;
        const MASS_CART: f32 = 1.0;
        const MASS_POLE: f32 = 0.1;
        const HALF_LENGTH: f32 = 0.5;
        const FORCE: f32 = 10.0;
        const TAU: f32 = 0.02;
        let [x, x_dot, theta, theta_dot] = self.state;
        let force = if action == 1 { FORCE } else { -FORCE };
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let total_mass = MASS_CART + MASS_POLE;
        let pole_mass_length = MASS_POLE * HALF_LENGTH;
        let temp = (force + pole_mass_length * theta_dot * theta_dot * sintheta) / total_mass;
        let theta_acc = (GRAVITY * sintheta - costheta * temp)
            / (HALF_LENGTH * (4. / 3. - MASS_POLE * costheta * costheta / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass;
        self.state = [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ];
        self.episode_return += 1.;
        self.state[0].abs() > 2.4 || self.state[2].abs() > 12. * PI / 180.
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as u32) as f32 / (1_u32 << 24) as f32
    }
}

fn policy(cx: Cx, observations: Tensor) -> NnResult<Tensor> {
    observations
        .apply(&cx.named_layer("hidden", Linear::new(64))?)?
        .tanh()?
        .apply(&cx.named_layer("policy", Linear::new(2))?)
}

fn training_model(
    cx: Cx,
    observations: Tensor,
    actions: Tensor,
    advantages: Tensor,
) -> NnResult<(Tensor, Tensor)> {
    let logits = policy(cx, observations)?;
    let loss = logits
        .cross_entropy_with_indices(&actions, 1)?
        .mul(&advantages)?
        .mean([0], false)?;
    Ok((loss, logits))
}

fn parameter_refs(parameters: &BTreeMap<String, Buffer>) -> impl Iterator<Item = (&str, &Buffer)> {
    parameters
        .iter()
        .map(|(name, value)| (name.as_str(), value))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.environments == 0 || args.horizon == 0 || args.iterations == 0 {
        return Err("iterations, environments, and horizon must be positive".into());
    }
    let batch = args.environments * args.horizon;
    let client = unsafe { Client::load(&args.plugin) }?;
    let mut compiler = Compiler::new(client.clone(), CacheLimits::default());

    let inference = Model::new(policy)
        .inputs(ModelInput::new([args.environments as i64, 4]))
        .trace()?
        .compile(&mut compiler)?;
    let mut training = Model::new(training_model)
        .inputs((
            ModelInput::new([batch as i64, 4]),
            ModelInput::new([batch as i64]).with_dtype(DType::I32),
            ModelInput::new([batch as i64]),
        ))
        .training()
        .trace()?;
    let selection = training.schema().select_all();
    let loss = training.outputs()[0].clone();
    let adam = prepare_model_adam(
        &mut training,
        &selection,
        &loss,
        AdamOptions {
            learning_rate: args.learning_rate,
            ..AdamOptions::default()
        },
    )?;
    let outputs = adam.outputs_with(&[loss]);
    let program = outputs.compile_stateful(&training, &mut compiler)?;
    let mut session = training.session(&program).build()?;
    let mut parameters = training
        .schema()
        .initialize(&client, args.seed)?
        .into_iter()
        .collect::<BTreeMap<_, _>>();

    let mut rng = Rng::new(args.seed);
    let mut environments = (0..args.environments)
        .map(|_| CartPole::reset(&mut rng))
        .collect::<Vec<_>>();
    let mut completed_returns = Vec::new();

    for iteration in 0..args.iterations {
        let mut observations = Vec::with_capacity(batch * 4);
        let mut actions = Vec::with_capacity(batch);
        let rewards = vec![1_f32; batch];
        let mut terminals = vec![false; batch];

        for time in 0..args.horizon {
            let step_observations = environments
                .iter()
                .flat_map(|environment| environment.state)
                .collect::<Vec<_>>();
            let input = client.buffer(&[args.environments as i64, 4], &step_observations)?;
            let logits: Buffer = inference.run(&input, parameter_refs(&parameters))?;
            let logits = logits.to_vec::<f32>()?;
            observations.extend_from_slice(&step_observations);
            for (environment_index, environment) in environments.iter_mut().enumerate() {
                let row = &logits[environment_index * 2..environment_index * 2 + 2];
                let probability_one = 1. / (1. + (row[0] - row[1]).exp());
                let action = usize::from(rng.uniform() < probability_one);
                actions.push(action as i32);
                let index = time * args.environments + environment_index;
                let terminal = environment.step(action);
                terminals[index] = terminal;
                if terminal {
                    completed_returns.push(environment.episode_return);
                    *environment = CartPole::reset(&mut rng);
                }
            }
        }

        let mut returns = vec![0_f32; batch];
        let mut carry = vec![0_f32; args.environments];
        for time in (0..args.horizon).rev() {
            for (environment, carry) in carry.iter_mut().enumerate() {
                let index = time * args.environments + environment;
                if terminals[index] {
                    *carry = 0.;
                }
                *carry = rewards[index] + args.gamma * *carry;
                returns[index] = *carry;
            }
        }
        let mean = returns.iter().sum::<f32>() / batch as f32;
        let variance = returns
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / batch as f32;
        let inverse_std = 1. / (variance + 1e-8).sqrt();
        for value in &mut returns {
            *value = (*value - mean) * inverse_std;
        }

        let observation_buffer = client.buffer(&[batch as i64, 4], &observations)?;
        let action_buffer = client.buffer(&[batch as i64], &actions)?;
        let advantage_buffer = client.buffer(&[batch as i64], &returns)?;
        let arguments = training.bind(
            (&observation_buffer, &action_buffer, &advantage_buffer),
            parameter_refs(&parameters),
        )?;
        let result = session.run(arguments.as_slice())?;
        let loss_value = result[0].to_vec::<f32>()?[0];
        outputs.commit(result, &mut parameters)?;

        if iteration % 10 == 0 || iteration + 1 == args.iterations {
            let recent = completed_returns.iter().rev().take(100);
            let (sum, count) = recent.fold((0., 0), |(sum, count), value| (sum + value, count + 1));
            println!(
                "iteration {iteration:>3}: loss {loss_value:>8.4}, recent return {:>6.1}, episodes {}",
                sum / count.max(1) as f32,
                completed_returns.len()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartpole_terminates_after_sustained_force() {
        let mut environment = CartPole::reset(&mut Rng::new(1));
        assert!((0..500).any(|_| environment.step(1)));
        assert!(environment.episode_return > 0.);
    }

    #[test]
    fn inference_and_training_share_parameter_schema() {
        let inference = Model::new(policy)
            .inputs(ModelInput::new([4, 4]))
            .trace()
            .unwrap();
        let training = Model::new(training_model)
            .inputs((
                ModelInput::new([16, 4]),
                ModelInput::new([16]).with_dtype(DType::I32),
                ModelInput::new([16]),
            ))
            .training()
            .trace()
            .unwrap();
        let inference_parameters = inference
            .schema()
            .parameters()
            .iter()
            .map(|parameter| (parameter.path(), parameter.shape()))
            .collect::<Vec<_>>();
        let training_parameters = training
            .schema()
            .parameters()
            .iter()
            .map(|parameter| (parameter.path(), parameter.shape()))
            .collect::<Vec<_>>();
        assert_eq!(inference_parameters, training_parameters);
        assert_eq!(inference.outputs()[0].shape(), [4, 2]);
        assert_eq!(training.outputs()[0].shape(), []);
    }
}
