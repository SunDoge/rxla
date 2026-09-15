//! End-to-end Stable Diffusion inference with device-resident PNDM state.
use clap::{Parser, ValueEnum};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use regex::Regex;
use rxla_core::{Buffer, CacheLimits, CacheStats, Client, Compiler, Executable, Tensor};
use rxla_models::{
    AutoencoderKlDecoderConfig, ClipTextConfig, PndmSampleSource, PndmScheduler, UnetConfig,
    autoencoder_kl_decoder, clip_text_encoder, unet,
};
use rxla_nn::{AppliedModel, Cx, ParamSchema, apply, init};
use rxla_safetensors::{SafeTensors, SchemaBuffers};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::{Path, PathBuf},
    rc::Rc,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Parser)]
#[command(about = "Run end-to-end Stable Diffusion inference")]
struct Args {
    /// Trusted PJRT plugin shared library.
    plugin: PathBuf,
    /// Diffusers model directory.
    model: PathBuf,
    #[arg(default_value = "a photo of an astronaut riding a horse")]
    prompt: String,
    #[arg(default_value_t = 20, value_parser = parse_positive_usize)]
    steps: usize,
    #[arg(default_value_t = 7.5, value_parser = parse_finite_f32)]
    guidance: f32,
    #[arg(default_value_t = 1)]
    warmup: usize,
    #[arg(default_value_t = 3, value_parser = parse_positive_usize)]
    iterations: usize,
    image: Option<PathBuf>,
    output: Option<PathBuf>,
    latent_output: Option<PathBuf>,
    /// Model architecture. Use sd-v1 for real Stable Diffusion 1.x weights.
    #[arg(long, value_enum, default_value_t = ModelPreset::Tiny)]
    preset: ModelPreset,
    /// Floating-point compute policy; the model ABI remains F32.
    #[arg(long, value_enum, default_value_t = ComputePrecision::F32)]
    precision: ComputePrecision,
    /// Private directory for trusted native PJRT executable artifacts.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelPreset {
    Tiny,
    SdV1,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComputePrecision {
    F32,
    F16,
    Bf16,
}

impl ModelPreset {
    fn clip(self) -> ClipTextConfig {
        match self {
            Self::Tiny => ClipTextConfig::tiny(),
            Self::SdV1 => ClipTextConfig::stable_diffusion_v1(),
        }
    }

    fn unet(self) -> UnetConfig {
        match self {
            Self::Tiny => UnetConfig::tiny(),
            Self::SdV1 => UnetConfig::stable_diffusion_v1(),
        }
    }

    fn vae(self) -> AutoencoderKlDecoderConfig {
        match self {
            Self::Tiny => AutoencoderKlDecoderConfig::tiny(),
            Self::SdV1 => AutoencoderKlDecoderConfig::stable_diffusion(),
        }
    }

    fn timestep_width(self) -> usize {
        match self {
            Self::Tiny => 32,
            Self::SdV1 => 320,
        }
    }

    fn context_width(self) -> i64 {
        match self {
            Self::Tiny => 32,
            Self::SdV1 => 768,
        }
    }

    fn image_size(self) -> usize {
        match self {
            Self::Tiny => 128,
            Self::SdV1 => 512,
        }
    }
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("expected a positive integer: {error}"))?;
    (value > 0)
        .then_some(value)
        .ok_or_else(|| "value must be positive".into())
}

fn parse_finite_f32(value: &str) -> std::result::Result<f32, String> {
    let value = value
        .parse::<f32>()
        .map_err(|error| format!("expected a number: {error}"))?;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| "value must be finite".into())
}

/// Exact CLIP BPE for the supplied vocabulary/merges. `tokenizers`' generic
/// ByteLevel pre-tokenizer follows GPT-2's leading-space convention, whereas
/// CLIP regex-splits before byte encoding and suffixes each word with `</w>`.
struct ClipTokenizer {
    vocabulary: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    pattern: Regex,
}

impl ClipTokenizer {
    fn open(directory: &Path) -> Result<Self> {
        let vocabulary =
            serde_json::from_reader(std::fs::File::open(directory.join("vocab.json"))?)?;
        let merges = std::fs::read_to_string(directory.join("merges.txt"))?
            .lines()
            .skip(1)
            .filter_map(|line| line.split_once(' '))
            .enumerate()
            .map(|(rank, (left, right))| ((left.into(), right.into()), rank))
            .collect();
        let visible: HashSet<u32> = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .map(u32::from)
            .collect();
        let mut next = 256_u32;
        let byte_encoder = std::array::from_fn(|byte| {
            let codepoint = if visible.contains(&(byte as u32)) {
                byte as u32
            } else {
                let codepoint = next;
                next += 1;
                codepoint
            };
            char::from_u32(codepoint).expect("CLIP byte encoder uses valid Unicode")
        });
        Ok(Self {
            vocabulary,
            merges,
            byte_encoder,
            pattern: Regex::new(
                r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            )?,
        })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token
            .chars()
            .map(|character| character.to_string())
            .collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        while let Some((left, right)) = word
            .windows(2)
            .filter_map(|pair| {
                let key = (pair[0].clone(), pair[1].clone());
                self.merges.get(&key).map(|&rank| (rank, key))
            })
            .min_by_key(|(rank, _)| *rank)
            .map(|(_, pair)| pair)
        {
            let mut merged = Vec::with_capacity(word.len());
            let mut index = 0;
            while index < word.len() {
                if index + 1 < word.len() && word[index] == left && word[index + 1] == right {
                    merged.push(format!("{left}{right}"));
                    index += 2;
                } else {
                    merged.push(word[index].clone());
                    index += 1;
                }
            }
            word = merged;
        }
        word
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let mut result = Vec::new();
        let normalized = text.to_lowercase();
        for matched in self.pattern.find_iter(&normalized) {
            let byte_encoded: String = matched
                .as_str()
                .as_bytes()
                .iter()
                .map(|byte| self.byte_encoder[*byte as usize])
                .collect();
            for piece in self.bpe(&byte_encoded) {
                result.push(i32::try_from(*self.vocabulary.get(&piece).unwrap_or(&1))?);
            }
        }
        Ok(result)
    }
}

fn tokenize(tokenizer: &ClipTokenizer, text: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer.encode(text)?;
    let mut ids = Vec::with_capacity(77);
    ids.push(0);
    ids.extend(encoding.into_iter().take(75));
    ids.push(1);
    ids.resize(77, 1);
    Ok(ids)
}

fn timestep_embedding(timestep: i64, width: usize) -> Vec<f32> {
    let half = width / 2;
    let angles: Vec<_> = (0..half)
        .map(|index| (-10_000_f32.ln() * index as f32 / half as f32).exp() * timestep as f32)
        .collect();
    angles
        .iter()
        .map(|value| value.cos())
        .chain(angles.iter().map(|value| value.sin()))
        .collect()
}

fn coefficient(coefficients: &Tensor, index: i64, shape: &[i64]) -> rxla_core::Result<Tensor> {
    coefficients.narrow(0, index, 1)?.broadcast_to(shape)
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn write_png(path: &Path, image: &[f32], height: usize, width: usize) -> Result<()> {
    let pixels = image
        .iter()
        .map(|value| (value * 255.).round().clamp(0., 255.) as u8)
        .collect::<Vec<_>>();
    image::save_buffer(
        path,
        &pixels,
        u32::try_from(width)?,
        u32::try_from(height)?,
        image::ColorType::Rgb8,
    )?;
    Ok(())
}

fn diffusers_vae_mapping(schema: &ParamSchema) -> HashMap<String, String> {
    schema
        .parameters()
        .iter()
        .map(|parameter| {
            let path = parameter.path().to_owned();
            let source = path
                .replace(
                    "decoder.mid_block.attentions.0.to_q.",
                    "decoder.mid_block.attentions.0.query.",
                )
                .replace(
                    "decoder.mid_block.attentions.0.to_k.",
                    "decoder.mid_block.attentions.0.key.",
                )
                .replace(
                    "decoder.mid_block.attentions.0.to_v.",
                    "decoder.mid_block.attentions.0.value.",
                )
                .replace(
                    "decoder.mid_block.attentions.0.to_out.0.",
                    "decoder.mid_block.attentions.0.proj_attn.",
                );
            (path, source)
        })
        .collect()
}

struct Stage {
    model: AppliedModel,
    executable: Rc<Executable>,
    weights: SchemaBuffers,
}

impl Stage {
    fn execute(&self, inputs: &[&Buffer]) -> rxla_nn::Result<Vec<Buffer>> {
        let bindings = self.weights.bindings();
        let arguments = self.model.bind(inputs, bindings)?;
        Ok(self.executable.execute(arguments.as_slice())?)
    }
}

struct Pipeline {
    clip: Stage,
    denoise: Stage,
    vae: Stage,
    tokens: Buffer,
    initial: Rc<Buffer>,
    zero: Rc<Buffer>,
    timesteps: Vec<Rc<Buffer>>,
    coefficients: Vec<Rc<Buffer>>,
    scheduler: PndmScheduler,
}

impl Pipeline {
    fn evaluate(&mut self) -> Result<(Buffer, Duration, Duration, Duration)> {
        let start = Instant::now();
        let context = Rc::new(self.clip.execute(&[&self.tokens])?.remove(0));
        let clip_time = start.elapsed();

        let denoise_bindings = self.denoise.weights.bindings();
        let denoise_parameters = self.denoise.model.bind_parameters(denoise_bindings)?;
        let start = Instant::now();
        let mut current = self.initial.clone();
        let initial = current.clone();
        let mut history: Vec<Rc<Buffer>> = Vec::with_capacity(4);
        for index in 0..self.scheduler.timesteps().len() {
            let plan = self.scheduler.step(index)?;
            let update_sample = match plan.sample_source {
                PndmSampleSource::Current => &current,
                PndmSampleSource::Initial => &initial,
            };
            let history_inputs: Vec<_> = (0..4)
                .map(|slot| history.get(slot).unwrap_or(&self.zero))
                .collect();
            let arguments = denoise_parameters.bind(&[
                current.as_ref(),
                update_sample.as_ref(),
                self.timesteps[index].as_ref(),
                context.as_ref(),
                history_inputs[0].as_ref(),
                history_inputs[1].as_ref(),
                history_inputs[2].as_ref(),
                history_inputs[3].as_ref(),
                self.coefficients[index].as_ref(),
            ])?;
            let mut outputs = self.denoise.executable.execute(arguments.as_slice())?;
            current = Rc::new(outputs.remove(0));
            let model_output = Rc::new(outputs.remove(0));
            if plan.retain_model_output {
                history.insert(0, model_output);
                history.truncate(4);
            }
        }
        let denoise_time = start.elapsed();
        let start = Instant::now();
        let image = self.vae.execute(&[current.as_ref()])?.remove(0);
        let vae_time = start.elapsed();
        Ok((image, clip_time, denoise_time, vae_time))
    }
}

fn build(args: &Args) -> Result<(Pipeline, Duration, CacheStats)> {
    let scheduler = PndmScheduler::stable_diffusion(args.steps)?;
    let tokenizer = ClipTokenizer::open(&args.model.join("tokenizer"))?;
    let mut token_ids = tokenize(&tokenizer, "")?;
    token_ids.extend(tokenize(&tokenizer, &args.prompt)?);

    let client = unsafe { Client::load(&args.plugin) }?;
    let mut compiler =
        Compiler::new(client.clone(), CacheLimits::default()).with_f16_attention(true);
    compiler = match args.precision {
        ComputePrecision::F32 => compiler,
        ComputePrecision::F16 => compiler.with_compute_dtype(rxla_core::DType::F16),
        ComputePrecision::Bf16 => compiler.with_compute_dtype(rxla_core::DType::BF16),
    };
    #[cfg(feature = "disk-cache")]
    if let Some(directory) = &args.cache_dir {
        let cache = unsafe {
            rxla_core::DiskCache::new_for_client(
                directory,
                "rxla-stable-diffusion-f16-attention-v1",
                &client,
                4 * 1024 * 1024 * 1024,
            )?
        };
        compiler = compiler.with_disk_cache(cache);
    }
    #[cfg(not(feature = "disk-cache"))]
    if args.cache_dir.is_some() {
        return Err("--cache-dir requires rxla-safetensors/disk-cache".into());
    }
    let compile_start = Instant::now();

    let clip_config = args.preset.clip();
    let clip_model = |cx: &mut Cx| {
        let tokens = cx.input_dtype(&[2, 77], rxla_core::DType::I32)?;
        clip_text_encoder(cx, &tokens, &clip_config)
    };
    let (clip_schema, _) = init(clip_model)?;
    let clip_model = apply(&clip_schema, clip_model)?;
    let mut clip_checkpoint = SafeTensors::open(args.model.join("text_encoder/model.safetensors"))?;
    let clip_executable = clip_model.compile(&mut compiler)?;
    let clip_weights = clip_checkpoint.load_parameter_schema(&client, &clip_schema)?;
    let clip = Stage {
        model: clip_model,
        executable: clip_executable,
        weights: clip_weights,
    };

    let unet_config = args.preset.unet();
    let context_width = args.preset.context_width();
    let timestep_width = args.preset.timestep_width();
    let denoise_model = |cx: &mut Cx| {
        let model_sample = cx.input(&[1, 64, 64, 4])?;
        let update_sample = cx.input(&[1, 64, 64, 4])?;
        let timestep = cx.input(&[2, timestep_width as i64])?;
        let context = cx.input(&[2, 77, context_width])?;
        let history = (0..4)
            .map(|_| cx.input(&[1, 64, 64, 4]))
            .collect::<rxla_nn::Result<Vec<_>>>()?;
        let coefficients = cx.input(&[8])?;
        let doubled = Tensor::concatenate(&[model_sample.clone(), model_sample.clone()], 0)?;
        let noise = unet(cx, &doubled, &timestep, &context, &unet_config)?;
        let split = noise.split(0, &[1, 1])?;
        let guidance = coefficient(&coefficients, 7, split[0].shape())?;
        let guided = split[0].add(&split[1].sub(&split[0])?.mul(&guidance)?)?;
        let mut combined = guided.mul(&coefficient(&coefficients, 2, guided.shape())?)?;
        for (slot, previous) in history.iter().enumerate() {
            combined = combined.add(&previous.mul(&coefficient(
                &coefficients,
                slot as i64 + 3,
                previous.shape(),
            )?)?)?;
        }
        let next = update_sample
            .mul(&coefficient(&coefficients, 0, update_sample.shape())?)?
            .add(&combined.mul(&coefficient(&coefficients, 1, combined.shape())?)?)?;
        Ok([next, guided])
    };
    let (denoise_schema, _) = init(denoise_model)?;
    let denoise_model = apply(&denoise_schema, denoise_model)?;
    let mut unet_checkpoint =
        SafeTensors::open(args.model.join("unet/diffusion_pytorch_model.safetensors"))?;
    let denoise_executable = denoise_model.compile(&mut compiler)?;
    let denoise_weights = unet_checkpoint.load_parameter_schema(&client, &denoise_schema)?;
    let denoise = Stage {
        model: denoise_model,
        executable: denoise_executable,
        weights: denoise_weights,
    };

    let vae_config = args.preset.vae();
    let vae_model = |cx: &mut Cx| -> rxla_nn::Result<Tensor> {
        let latent = cx.input(&[1, 64, 64, 4])?;
        Ok(autoencoder_kl_decoder(cx, &latent, &vae_config)?
            .mul_scalar(0.5)?
            .add_scalar(0.5)?
            .clamp(0.0, 1.0)?)
    };
    let (vae_schema, _) = init(vae_model)?;
    let vae_model = apply(&vae_schema, vae_model)?;
    let mut vae_checkpoint =
        SafeTensors::open(args.model.join("vae/diffusion_pytorch_model.safetensors"))?;
    let vae_executable = vae_model.compile(&mut compiler)?;
    let vae_weights = match args.preset {
        ModelPreset::Tiny => vae_checkpoint.load_parameter_schema(&client, &vae_schema)?,
        ModelPreset::SdV1 => vae_checkpoint.load_parameter_schema_with_mapping(
            &client,
            &vae_schema,
            &diffusers_vae_mapping(&vae_schema),
        )?,
    };
    let vae = Stage {
        model: vae_model,
        executable: vae_executable,
        weights: vae_weights,
    };
    let compile_time = compile_start.elapsed();

    let tokens = client.buffer(&[2, 77], &token_ids)?;
    println!("prompt_token_ids={:?}", &token_ids[77..]);

    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let initial_values: Vec<f32> = (0..64 * 64 * 4)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    if let Some(path) = &args.latent_output {
        write_f32(path, &initial_values)?;
    }
    let initial = Rc::new(client.buffer(&[1, 64, 64, 4], &initial_values)?);
    let zero = Rc::new(client.buffer(&[1, 64, 64, 4], &vec![0.; 64 * 64 * 4])?);
    let timesteps = scheduler
        .timesteps()
        .iter()
        .map(|&value| {
            let embedding = timestep_embedding(value, timestep_width);
            let doubled: Vec<_> = embedding.iter().chain(&embedding).copied().collect();
            Ok(Rc::new(
                client.buffer(&[2, timestep_width as i64], &doubled)?,
            ))
        })
        .collect::<Result<_>>()?;
    let coefficients = (0..scheduler.timesteps().len())
        .map(|index| {
            Ok(Rc::new(client.buffer(
                &[8],
                &scheduler.step(index)?.tensor_coefficients(args.guidance)?,
            )?))
        })
        .collect::<Result<_>>()?;
    let cache_stats = compiler.stats();
    Ok((
        Pipeline {
            clip,
            denoise,
            vae,
            tokens,
            initial,
            zero,
            timesteps,
            coefficients,
            scheduler,
        },
        compile_time,
        cache_stats,
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (mut pipeline, compile_time, cache_stats) = build(&args)?;
    for _ in 0..args.warmup {
        pipeline.evaluate()?;
    }
    let mut totals = Vec::with_capacity(args.iterations);
    let mut clip_total = Duration::ZERO;
    let mut denoise_total = Duration::ZERO;
    let mut vae_total = Duration::ZERO;
    let mut image = None;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let (result, clip, denoise, vae) = pipeline.evaluate()?;
        totals.push(start.elapsed().as_secs_f64() * 1e3);
        clip_total += clip;
        denoise_total += denoise;
        vae_total += vae;
        image = Some(result);
    }
    let download_start = Instant::now();
    let image = image
        .expect("iterations validated positive")
        .to_vec::<f32>()?;
    let download_time = download_start.elapsed();
    if let Some(path) = &args.image {
        let size = args.preset.image_size();
        write_png(path, &image, size, size)?;
    }
    if let Some(path) = &args.output {
        write_f32(path, &image)?;
    }
    totals.sort_by(f64::total_cmp);
    let count = args.iterations as f64;
    println!(
        "steps={} compile_ms={:.3} compile_misses={} disk_cache_hits={} disk_cache_read_errors={} disk_cache_write_errors={} mean_ms={:.3} p50_ms={:.3} p95_ms={:.3} clip_mean_ms={:.3} denoise_mean_ms={:.3} vae_mean_ms={:.3} download_ms={:.3} checksum={:.9}",
        args.steps,
        compile_time.as_secs_f64() * 1e3,
        cache_stats.misses,
        cache_stats.disk_hits,
        cache_stats.disk_read_errors,
        cache_stats.disk_write_errors,
        totals.iter().sum::<f64>() / count,
        totals[((totals.len() - 1) as f64 * 0.5).round() as usize],
        totals[((totals.len() - 1) as f64 * 0.95).round() as usize],
        clip_total.as_secs_f64() * 1e3 / count,
        denoise_total.as_secs_f64() * 1e3 / count,
        vae_total.as_secs_f64() * 1e3 / count,
        download_time.as_secs_f64() * 1e3,
        image.iter().map(|&value| f64::from(value)).sum::<f64>(),
    );
    Ok(())
}
