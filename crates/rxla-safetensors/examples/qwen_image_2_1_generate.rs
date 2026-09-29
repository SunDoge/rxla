//! Qwen-Image-2.1 W8 text-to-image generation from cached Qwen3-VL embeddings.

use clap::Parser;
use image::{ImageBuffer, Rgba};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rxla_core::{Buffer, CacheLimits, Client, ClientOptions, Compiler, Memory, Tensor};
use rxla_models::qwen_image_2_1::{
    QwenImage21Config, QwenImage21FlowScheduler, qwen_image_2_1_denoise, qwen_image_2_1_euler_step,
    qwen_image_2_1_vae_decoder, qwen_image_2_1_vae_decoder_raw,
};
use rxla_nn::{BoundModel, BufferProgram, Cx, MicroBatchPlan, Model, ModelInput};
use rxla_safetensors::{ResidentBoundModel, SafeTensors, WeightResidencyPool};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Generate an RGBA image with Qwen-Image-2.1 W8")]
struct Args {
    /// Trusted CUDA PJRT plugin.
    plugin: PathBuf,
    /// RXLA grouped-W8 transformer safetensors.
    transformer: PathBuf,
    /// Official VAE safetensors.
    vae: PathBuf,
    /// Official VAE config.json containing latent mean/std.
    vae_config: PathBuf,
    /// Safetensors containing `prompt_embeds` with shape [1, tokens, 4096].
    conditioning: PathBuf,
    /// Output PNG.
    output: PathBuf,
    #[arg(long, default_value_t = 512)]
    width: u32,
    #[arg(long, default_value_t = 512)]
    height: u32,
    #[arg(long, default_value_t = 40)]
    steps: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of requests executed in the same initialized process. Only the
    /// final request is encoded; earlier requests measure steady-state serving.
    #[arg(long, default_value_t = 1)]
    requests: usize,
    /// Number of images generated together by the transformer and VAE.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Maximum number of latents decoded by the VAE at once. The transformer
    /// can use a larger batch without requiring one large contiguous VAE
    /// allocation.
    #[arg(long, default_value_t = 4)]
    vae_batch_size: usize,
    /// VAE tile side in output pixels. Larger images are decoded as overlapping tiles.
    #[arg(long, default_value_t = 1024)]
    vae_tile: u32,
    /// VAE tile overlap in output pixels.
    #[arg(long, default_value_t = 128)]
    vae_overlap: u32,
    /// PJRT allocator fraction; lower this when sharing the GPU.
    #[arg(long)]
    memory_fraction: Option<f32>,
    /// Reserve the complete allocator pool at client creation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    preallocate: bool,
    /// Put all denoising steps in one StableHLO while loop. The asynchronous
    /// host enqueue path remains faster on the currently tested CUDA plugin.
    #[arg(long)]
    device_loop: bool,
    /// Persistent serialized PJRT executable cache.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Retain transformer weights in PJRT pinned-host memory and prefetch them
    /// only for denoising. Intended for multi-component serving under a GPU
    /// memory budget.
    #[arg(long)]
    offload_transformer: bool,
    /// Device allocator space kept outside the weight pool for activations and
    /// executable temporaries.
    #[arg(long, default_value_t = 2048)]
    offload_headroom_mib: usize,
}

#[derive(Deserialize)]
struct VaeConfig {
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

fn compiler(client: &Client, cache_dir: Option<&PathBuf>, _namespace: &str) -> Result<Compiler> {
    let compiler = Compiler::new(client.clone(), CacheLimits::default());
    #[cfg(feature = "disk-cache")]
    let compiler = if let Some(directory) = cache_dir {
        let cache = unsafe {
            rxla_core::DiskCache::new_for_client(
                directory,
                _namespace,
                client,
                4 * 1024 * 1024 * 1024,
            )?
        };
        compiler.with_disk_cache(cache)
    } else {
        compiler
    };
    #[cfg(not(feature = "disk-cache"))]
    if cache_dir.is_some() {
        return Err("--cache-dir requires rxla-safetensors/disk-cache".into());
    }
    Ok(compiler)
}

struct DenoiseOptions<'a> {
    checkpoint_path: &'a PathBuf,
    conditioning_path: &'a PathBuf,
    latent_height: i64,
    latent_width: i64,
    steps: usize,
    batch_size: usize,
    device_loop: bool,
    cache_dir: Option<&'a PathBuf>,
    offload_transformer: bool,
    offload_headroom_bytes: usize,
}

fn cuda_offload_memories(client: &Client) -> Result<(Memory, Memory)> {
    let memories = client.memories()?;
    let host = memories
        .iter()
        .find(|memory| {
            memory
                .info()
                .is_ok_and(|info| info.kind.contains("pinned_host"))
        })
        .cloned()
        .ok_or("PJRT client does not expose pinned-host memory")?;
    let device = client.default_memory()?;
    Ok((host, device))
}

enum DenoiseRunner {
    Device(BoundModel),
    Offloaded(ResidentBoundModel),
}

struct Denoiser {
    client: Client,
    runner: DenoiseRunner,
    context: Buffer,
    schedule: QwenImage21FlowScheduler,
    batch: i64,
    batch_size: usize,
    image_tokens: i64,
    device_loop: bool,
}

impl Denoiser {
    fn run(&self, seed: u64) -> Result<(Buffer, std::time::Duration)> {
        let mut random = ChaCha8Rng::seed_from_u64(seed);
        let initial = (0..self.batch * self.image_tokens * 64)
            .map(|_| StandardNormal.sample(&mut random))
            .collect::<Vec<f32>>();
        let mut sample = self
            .client
            .buffer(&[self.batch, self.image_tokens, 64], &initial)?;
        let execute_start = Instant::now();
        if self.device_loop {
            sample = match &self.runner {
                DenoiseRunner::Device(runner) => runner.run((&sample, &self.context))?,
                DenoiseRunner::Offloaded(runner) => runner.run((&sample, &self.context))?,
            };
        } else {
            for index in 0..self.schedule.len() {
                let step = self.schedule.step(index)?;
                let timestep = self
                    .client
                    .buffer(&[self.batch], &vec![step.timestep; self.batch_size])?;
                let step_size = self
                    .client
                    .buffer(&[self.batch], &vec![step.step_size; self.batch_size])?;
                sample = match &self.runner {
                    DenoiseRunner::Device(runner) => {
                        runner.run((&sample, &timestep, &step_size, &self.context))?
                    }
                    DenoiseRunner::Offloaded(runner) => {
                        runner.run((&sample, &timestep, &step_size, &self.context))?
                    }
                };
            }
        }
        Ok((sample, execute_start.elapsed()))
    }
}

fn prepare_denoiser(client: &Client, options: &DenoiseOptions<'_>) -> Result<Denoiser> {
    let mut conditioning = SafeTensors::open(options.conditioning_path)?;
    let prompt = conditioning.read_f32("prompt_embeds")?;
    if prompt.shape.len() != 3 || prompt.shape[0] != 1 || prompt.shape[2] != 4096 {
        return Err("prompt_embeds must have shape [1, tokens, 4096]".into());
    }
    let context_tokens = prompt.shape[1];
    let batch = i64::try_from(options.batch_size)?;
    let image_tokens = options.latent_height * options.latent_width;
    let config = QwenImage21Config::official_w8(options.latent_height, options.latent_width);
    let schedule = QwenImage21FlowScheduler::new(options.steps, image_tokens as usize)?;
    let trace_start = Instant::now();
    let model = if options.device_loop {
        Model::new(|cx: Cx, sample, context| {
            qwen_image_2_1_denoise(cx, &sample, &context, &schedule, &config)
        })
        .inputs((
            ModelInput::new([batch, image_tokens, 64]),
            ModelInput::new([batch, context_tokens, 4096]),
        ))
        .trace()?
    } else {
        Model::new(|cx: Cx, sample, timestep, step_size, context| {
            qwen_image_2_1_euler_step(cx, &sample, &timestep, &step_size, &context, &config)
        })
        .inputs((
            ModelInput::new([batch, image_tokens, 64]),
            ModelInput::new([batch]),
            ModelInput::new([batch]),
            ModelInput::new([batch, context_tokens, 4096]),
        ))
        .trace()?
    };
    let trace_time = trace_start.elapsed();
    let load_start = Instant::now();
    let mut checkpoint = SafeTensors::open(options.checkpoint_path)?;
    checkpoint.require_metadata(&[
        ("format", "rxla-qwen-image-2.1-w8-v1"),
        ("group_size", "64"),
    ])?;
    let (offloaded, resident) = if options.offload_transformer {
        let (host, _) = cuda_offload_memories(client)?;
        let buffers = checkpoint.load_parameter_schema(&host, model.schema())?;
        (Some(buffers.offload_to(&host)?), None)
    } else {
        (
            None,
            Some(checkpoint.load_parameter_schema(client, model.schema())?),
        )
    };
    let load_time = load_start.elapsed();
    let residency_pool = if let Some(offloaded) = &offloaded {
        let (_, device) = cuda_offload_memories(client)?;
        Some(
            WeightResidencyPool::new(device, offloaded.payload_bytes())
                .with_allocator_headroom(options.offload_headroom_bytes),
        )
    } else {
        None
    };
    let prefetch_launch_start = Instant::now();
    let pending_prefetch = match (&offloaded, &residency_pool) {
        (Some(offloaded), Some(pool)) => Some(pool.prefetch_async(offloaded.clone())),
        (None, None) => None,
        _ => return Err("invalid transformer residency pool state".into()),
    };
    let prefetch_launch_time = prefetch_launch_start.elapsed();
    let compile_start = Instant::now();
    let mut compiler =
        compiler(client, options.cache_dir, "qwen-image-2.1-dit-v1")?.with_cudnn_attention(true);
    if !options.device_loop {
        compiler = compiler.with_compute_dtype(rxla_core::DType::BF16);
    }
    let compiled = model.compile(&mut compiler)?;
    let compile_time = compile_start.elapsed();
    let prefetch_wait_start = Instant::now();
    let prefetched = pending_prefetch.map(|pending| pending.wait()).transpose()?;
    if let Some(prefetched) = &prefetched {
        prefetched.block_until_ready()?;
    }
    let prefetch_wait_time = prefetch_wait_start.elapsed();
    let runner = match (prefetched, resident) {
        (Some(prefetched), None) => DenoiseRunner::Offloaded(prefetched.bind(&compiled)?),
        (None, Some(resident)) => {
            DenoiseRunner::Device(compiled.bind_parameters(resident.bindings())?)
        }
        _ => return Err("invalid transformer residency state".into()),
    };
    let context_values = prompt.values.repeat(options.batch_size);
    let context = client.buffer(&[batch, context_tokens, 4096], &context_values)?;
    eprintln!(
        "denoiser-init mode={} residency={} batch={} trace={trace_time:.3?} weights={load_time:.3?} compile={compile_time:.3?} prefetch-launch={prefetch_launch_time:.3?} prefetch-tail={prefetch_wait_time:.3?} steps={}",
        if options.device_loop {
            "device-loop"
        } else {
            "host-enqueue"
        },
        if options.offload_transformer {
            "pinned-host"
        } else {
            "device"
        },
        options.batch_size,
        options.steps,
    );
    Ok(Denoiser {
        client: client.clone(),
        runner,
        context,
        schedule,
        batch,
        batch_size: options.batch_size,
        image_tokens,
        device_loop: options.device_loop,
    })
}

struct DecodeOptions<'a> {
    checkpoint_path: &'a Path,
    latent_height: i64,
    latent_width: i64,
    batch_size: usize,
    max_batch_size: usize,
    cache_dir: Option<&'a PathBuf>,
}

fn prepare_vae_bridge(
    client: &Client,
    mean: &[f32],
    std: &[f32],
    options: &DecodeOptions<'_>,
) -> Result<(BufferProgram, Vec<usize>)> {
    let image_tokens = options.latent_height * options.latent_width;
    let plan = MicroBatchPlan::new(options.batch_size, options.max_batch_size)?;
    let model = Model::new(|cx: Cx, latent: Tensor| -> rxla_nn::Result<Vec<Tensor>> {
        let mean = cx
            .constant([1, 1, 64], mean)?
            .broadcast_to(latent.shape())?;
        let std = cx.constant([1, 1, 64], std)?.broadcast_to(latent.shape())?;
        let normalized = latent.mul(&std)?.add(&mean)?;
        plan.split_zero_padded(&normalized, 0)
    })
    .inputs(ModelInput::new([
        i64::try_from(options.batch_size)?,
        image_tokens,
        64,
    ]))
    .trace()?;
    let compiled = model.compile(&mut compiler(
        client,
        options.cache_dir,
        "qwen-image-2.1-vae-input-v1",
    )?)?;
    let runner = compiled.bind_buffer_program()?;
    let actual_batches = plan.chunks().iter().map(|chunk| chunk.length()).collect();
    Ok((runner, actual_batches))
}

struct VaeDecoder {
    bridge: BufferProgram,
    decoder: BoundModel,
    actual_batches: Vec<usize>,
    values_per_image: usize,
    batch_size: usize,
}

impl VaeDecoder {
    fn run(&self, latent: &Buffer) -> Result<Vec<f32>> {
        let latent_batches: Vec<Buffer> = self.bridge.run(latent)?;
        let mut images = Vec::with_capacity(self.values_per_image * self.batch_size);
        for (latent, &actual_batch) in latent_batches.iter().zip(&self.actual_batches) {
            let image: Buffer = self.decoder.run(latent)?;
            let values = image.to_vec::<f32>()?;
            images.extend_from_slice(&values[..self.values_per_image * actual_batch]);
        }
        Ok(images)
    }
}

fn prepare_vae_decoder(
    client: &Client,
    mean: &[f32],
    std: &[f32],
    options: &DecodeOptions<'_>,
) -> Result<VaeDecoder> {
    let (bridge, actual_batches) = prepare_vae_bridge(client, mean, std, options)?;
    let image_tokens = options.latent_height * options.latent_width;
    let decode_batch_size = options.batch_size.min(options.max_batch_size);
    let model = Model::new(|cx: Cx, latent| {
        qwen_image_2_1_vae_decoder(cx, &latent, options.latent_height, options.latent_width)
    })
    .inputs(ModelInput::new([
        i64::try_from(decode_batch_size)?,
        image_tokens,
        64,
    ]))
    .trace()?;
    let mut checkpoint = SafeTensors::open(options.checkpoint_path)?;
    let parameters = checkpoint.load_parameter_schema(client, model.schema())?;
    let compiled = model.compile(&mut compiler(
        client,
        options.cache_dir,
        "qwen-image-2.1-vae-v1",
    )?)?;
    let decoder = compiled.bind_parameters(parameters.bindings())?;
    let values_per_image =
        usize::try_from(options.latent_height * 16 * options.latent_width * 16 * 4)?;
    Ok(VaeDecoder {
        bridge,
        decoder,
        actual_batches,
        values_per_image,
        batch_size: options.batch_size,
    })
}

fn tile_starts(length: usize, tile: usize, stride: usize) -> Vec<usize> {
    if length <= tile {
        return vec![0];
    }
    let mut starts = (0..=length - tile).step_by(stride).collect::<Vec<_>>();
    let final_start = length - tile;
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    starts
}

fn axis_weight(position: usize, index: usize, starts: &[usize], tile: usize) -> f32 {
    let mut weight = 1.0_f32;
    if index > 0 {
        let overlap = starts[index - 1] + tile - starts[index];
        if position < overlap {
            weight = weight.min((position + 1) as f32 / (overlap + 1) as f32);
        }
    }
    if index + 1 < starts.len() {
        let overlap = starts[index] + tile - starts[index + 1];
        if position >= tile - overlap {
            weight = weight.min((tile - position) as f32 / (overlap + 1) as f32);
        }
    }
    weight
}

#[allow(clippy::too_many_arguments)]
fn decode_tiled(
    client: &Client,
    checkpoint_path: &PathBuf,
    latent: &[f32],
    latent_height: usize,
    latent_width: usize,
    tile_height: usize,
    tile_width: usize,
    stride_height: usize,
    stride_width: usize,
    cache_dir: Option<&PathBuf>,
) -> Result<Vec<f32>> {
    const CHANNELS: usize = 64;
    const SCALE: usize = 16;
    const OUTPUT_CHANNELS: usize = 4;
    let tile_height_i64 = i64::try_from(tile_height)?;
    let tile_width_i64 = i64::try_from(tile_width)?;
    let tile_tokens_i64 = i64::try_from(tile_height * tile_width)?;
    let model = Model::new(|cx: Cx, latent| {
        qwen_image_2_1_vae_decoder_raw(cx, &latent, tile_height_i64, tile_width_i64)
    })
    .inputs(ModelInput::new([1, tile_tokens_i64, 64]))
    .trace()?;
    let mut checkpoint = SafeTensors::open(checkpoint_path)?;
    let parameters = checkpoint.load_parameter_schema(client, model.schema())?;
    let compiled = model.compile(&mut compiler(
        client,
        cache_dir,
        "qwen-image-2.1-vae-tiled-v1",
    )?)?;
    let runner = compiled.bind_parameters(parameters.bindings())?;
    let y_starts = tile_starts(latent_height, tile_height, stride_height);
    let x_starts = tile_starts(latent_width, tile_width, stride_width);
    let output_height = latent_height * SCALE;
    let output_width = latent_width * SCALE;
    let mut accumulated = vec![0.0_f32; output_height * output_width * OUTPUT_CHANNELS];
    let mut weights = vec![0.0_f32; output_height * output_width];
    let output_tile_height = tile_height * SCALE;
    let output_tile_width = tile_width * SCALE;
    let output_y_starts = y_starts
        .iter()
        .map(|start| start * SCALE)
        .collect::<Vec<_>>();
    let output_x_starts = x_starts
        .iter()
        .map(|start| start * SCALE)
        .collect::<Vec<_>>();
    for (tile_y, &start_y) in y_starts.iter().enumerate() {
        for (tile_x, &start_x) in x_starts.iter().enumerate() {
            let mut tile = vec![0.0_f32; tile_height * tile_width * CHANNELS];
            for y in 0..tile_height {
                let source = ((start_y + y) * latent_width + start_x) * CHANNELS;
                let target = y * tile_width * CHANNELS;
                tile[target..target + tile_width * CHANNELS]
                    .copy_from_slice(&latent[source..source + tile_width * CHANNELS]);
            }
            let tile = client.buffer(&[1, i64::try_from(tile_height * tile_width)?, 64], &tile)?;
            let decoded: Buffer = runner.run(&tile)?;
            let decoded = decoded.to_vec::<f32>()?;
            for y in 0..output_tile_height {
                let wy = axis_weight(y, tile_y, &output_y_starts, output_tile_height);
                let output_y = start_y * SCALE + y;
                for x in 0..output_tile_width {
                    let wx = axis_weight(x, tile_x, &output_x_starts, output_tile_width);
                    let weight = wy * wx;
                    let output_x = start_x * SCALE + x;
                    let output_pixel = output_y * output_width + output_x;
                    let tile_pixel = y * output_tile_width + x;
                    weights[output_pixel] += weight;
                    for channel in 0..OUTPUT_CHANNELS {
                        accumulated[output_pixel * OUTPUT_CHANNELS + channel] +=
                            decoded[tile_pixel * OUTPUT_CHANNELS + channel] * weight;
                    }
                }
            }
            eprintln!(
                "vae_tile={}/{} origin={}x{}",
                tile_y * x_starts.len() + tile_x + 1,
                y_starts.len() * x_starts.len(),
                start_x * SCALE,
                start_y * SCALE,
            );
        }
    }
    for (pixel, &weight) in accumulated
        .as_chunks_mut::<OUTPUT_CHANNELS>()
        .0
        .iter_mut()
        .zip(&weights)
    {
        if weight == 0.0 {
            return Err("VAE tiling left an uncovered output pixel".into());
        }
        for channel in pixel {
            *channel /= weight;
        }
    }
    Ok(accumulated)
}

fn run(args: Args) -> Result<()> {
    if args.batch_size == 0 || args.vae_batch_size == 0 || args.requests == 0 {
        return Err("batch sizes and requests must be positive".into());
    }
    if args.width == 0
        || args.height == 0
        || !args.width.is_multiple_of(16)
        || !args.height.is_multiple_of(16)
    {
        return Err("width and height must be positive multiples of 16".into());
    }
    if args.vae_tile == 0
        || !args.vae_tile.is_multiple_of(16)
        || args.vae_overlap >= args.vae_tile
        || !args.vae_overlap.is_multiple_of(16)
    {
        return Err("VAE tile and overlap must be multiples of 16, with tile > overlap".into());
    }
    if args
        .memory_fraction
        .is_some_and(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
    {
        return Err("memory fraction must be finite and between zero and one".into());
    }
    let mut client_options = ClientOptions::new().set("preallocate", args.preallocate);
    if let Some(memory_fraction) = args.memory_fraction {
        client_options = client_options.set("memory_fraction", memory_fraction);
    }
    let client = unsafe { Client::load_with_options(&args.plugin, &client_options) }?;
    let initialization_start = Instant::now();
    let latent_height = i64::from(args.height / 16);
    let latent_width = i64::from(args.width / 16);
    let denoiser = prepare_denoiser(
        &client,
        &DenoiseOptions {
            checkpoint_path: &args.transformer,
            conditioning_path: &args.conditioning,
            latent_height,
            latent_width,
            steps: args.steps,
            batch_size: args.batch_size,
            device_loop: args.device_loop,
            cache_dir: args.cache_dir.as_ref(),
            offload_transformer: args.offload_transformer,
            offload_headroom_bytes: args
                .offload_headroom_mib
                .checked_mul(1024 * 1024)
                .ok_or("offload headroom exceeds usize")?,
        },
    )?;
    let vae_config: VaeConfig = serde_json::from_slice(&fs::read(&args.vae_config)?)?;
    if vae_config.latents_mean.len() != 64 || vae_config.latents_std.len() != 64 {
        return Err("VAE config must contain 64 latent means and standard deviations".into());
    }
    let decode_options = DecodeOptions {
        checkpoint_path: &args.vae,
        latent_height,
        latent_width,
        batch_size: args.batch_size,
        max_batch_size: args.vae_batch_size,
        cache_dir: args.cache_dir.as_ref(),
    };
    let tiled = args.width > args.vae_tile || args.height > args.vae_tile;
    if tiled && args.requests != 1 {
        return Err(
            "multiple persistent requests are not yet supported with tiled VAE decode".into(),
        );
    }
    let vae_decoder = if tiled {
        None
    } else {
        Some(prepare_vae_decoder(
            &client,
            &vae_config.latents_mean,
            &vae_config.latents_std,
            &decode_options,
        )?)
    };
    let initialization_time = initialization_start.elapsed();
    eprintln!("pipeline-init={initialization_time:.3?}");
    let mut final_values = None;
    let mut request_times = Vec::with_capacity(args.requests);
    for request in 0..args.requests {
        let request_start = Instant::now();
        let seed = args
            .seed
            .checked_add(u64::try_from(request)?)
            .ok_or("request seed overflow")?;
        let (latent, denoise_time) = denoiser.run(seed)?;
        let vae_start = Instant::now();
        let values = if let Some(decoder) = &vae_decoder {
            decoder.run(&latent)?
        } else {
            let mut latent = latent.to_vec::<f32>()?;
            for token in latent.as_chunks_mut::<64>().0 {
                for (channel, value) in token.iter_mut().enumerate() {
                    *value =
                        *value * vae_config.latents_std[channel] + vae_config.latents_mean[channel];
                }
            }
            let tile_height = usize::try_from((args.vae_tile / 16).min(args.height / 16))?;
            let tile_width = usize::try_from((args.vae_tile / 16).min(args.width / 16))?;
            let stride = usize::try_from((args.vae_tile - args.vae_overlap) / 16)?;
            let values_per_latent = usize::try_from(latent_height * latent_width * 64)?;
            let mut decoded = Vec::new();
            for sample in latent.chunks_exact(values_per_latent) {
                decoded.extend(decode_tiled(
                    &client,
                    &args.vae,
                    sample,
                    usize::try_from(latent_height)?,
                    usize::try_from(latent_width)?,
                    tile_height,
                    tile_width,
                    stride.min(tile_height),
                    stride.min(tile_width),
                    args.cache_dir.as_ref(),
                )?);
            }
            decoded
        };
        let vae_time = vae_start.elapsed();
        let request_time = request_start.elapsed();
        eprintln!(
            "request={request} denoise={denoise_time:.3?} vae={vae_time:.3?} total={request_time:.3?} images_per_second={:.3}",
            args.batch_size as f64 / request_time.as_secs_f64(),
        );
        request_times.push(request_time);
        final_values = Some(values);
    }
    let pixels = final_values
        .ok_or("no generation request executed")?
        .into_iter()
        .map(|value| ((value + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    let pixels_per_image = usize::try_from(args.width)? * usize::try_from(args.height)? * 4;
    let outputs = output_paths(&args.output, args.batch_size);
    if pixels.len() != pixels_per_image * args.batch_size {
        return Err("VAE output does not match requested batch dimensions".into());
    }
    for (pixels, output) in pixels.chunks_exact(pixels_per_image).zip(&outputs) {
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(args.width, args.height, pixels.to_vec())
            .ok_or("VAE output does not match requested dimensions")?;
        image.save(output)?;
    }
    println!(
        "batch={} requests={} pipeline_init={initialization_time:.3?} median_request={:.3?} steady_images_per_second={:.3} saved {}",
        args.batch_size,
        args.requests,
        median_duration(&request_times),
        args.batch_size as f64 / median_duration(&request_times).as_secs_f64(),
        outputs
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn median_duration(values: &[std::time::Duration]) -> std::time::Duration {
    let mut values = values.to_vec();
    values.sort_unstable();
    values[values.len() / 2]
}

fn output_paths(output: &Path, batch_size: usize) -> Vec<PathBuf> {
    if batch_size == 1 {
        return vec![output.to_path_buf()];
    }
    let parent = output.parent().unwrap_or_else(|| std::path::Path::new(""));
    let stem = output
        .file_stem()
        .unwrap_or(output.as_os_str())
        .to_string_lossy();
    let extension = output.extension().map(|value| value.to_string_lossy());
    (0..batch_size)
        .map(|index| {
            let name = match &extension {
                Some(extension) => format!("{stem}-{index:03}.{extension}"),
                None => format!("{stem}-{index:03}"),
            };
            parent.join(name)
        })
        .collect()
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feathered_tiles_cover_irregular_final_stride_with_unit_weight() {
        let starts = tile_starts(128, 64, 48);
        assert_eq!(starts, [0, 48, 64]);
        let mut weights = [0.0_f32; 128];
        for (index, &start) in starts.iter().enumerate() {
            for position in 0..64 {
                weights[start + position] += axis_weight(position, index, &starts, 64);
            }
        }
        for weight in weights {
            assert!((weight - 1.0).abs() < 1e-6, "partition weight {weight}");
        }
    }
}
