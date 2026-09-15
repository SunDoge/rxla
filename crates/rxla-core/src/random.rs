//! Explicit counter-based graph randomness, independent of host Initializer.
use crate::{Result, Tensor, elements, err};
use std::sync::Arc;
mod state;
pub use state::{DropoutSample, ThreefrySequence, ThreefryState};

/// A categorical draw with the category axis removed from both outputs.
pub struct CategoricalSample {
    pub indices: Tensor,
    /// F32 0/1 per row. Invalid rows return placeholder index zero.
    pub valid: Tensor,
}

/// Nucleus (top-p) sampling, returning original category IDs. Sorts descending
/// stably and retains the smallest prefix reaching probability p, including
/// its crossing category. Requires finite p in (0, 1]; p=1 retains all categories.
/// Uses F32 softmax/cumulative sums, so cutoff decisions near p can vary with
/// backend rounding. No exact real-arithmetic cutoff guarantee is made.
///
/// `bits` matches the full logits shape and is assigned in sorted rank order.
/// Gumbel sampling follows `categorical_from_bits`; every original logit is
/// validated, even if excluded. Finite values and -infinity masks are allowed;
/// NaN/+infinity/all-masked rows return valid=0 and placeholder index zero.
/// No RNG state, temperature scaling, or top-k restriction is implicit.
pub fn top_p_categorical_from_bits(
    logits: &Tensor,
    bits: &Tensor,
    p: f32,
    axis: usize,
) -> Result<CategoricalSample> {
    if !p.is_finite() || p <= 0. || p > 1. {
        return Err(err("top-p probability must be finite and in (0, 1]"));
    }
    if !Arc::ptr_eq(&logits.graph().0, &bits.graph().0) || logits.shape() != bits.shape() {
        return Err(err("top-p bits must match logits shape and graph"));
    }
    if axis >= logits.shape().len() || logits.shape()[axis] == 0 {
        return Err(err("top-p requires a valid nonempty category axis"));
    }
    let (sorted, candidates) = logits.topk(logits.shape()[axis] as usize, axis)?;
    let masked = logits
        .graph()
        .constant(&[], &[f32::NEG_INFINITY])?
        .broadcast_to(logits.shape())?;
    let filtered = if p == 1. {
        sorted
    } else {
        let cumulative = sorted.softmax(axis)?.cumsum(axis)?;
        // Shift prefix sums instead of subtracting the current probability:
        // cancellation could lose the preceding mass of a dominant category.
        let mut padding = vec![[0, 0]; logits.shape().len()];
        padding[axis] = [1, 0];
        let preceding = cumulative
            .narrow(axis, 0, logits.shape()[axis] - 1)?
            .pad(&padding, 0.)?;
        let keep = preceding.lt_mask(
            &logits
                .graph()
                .constant(&[], &[p])?
                .broadcast_to(logits.shape())?,
        )?;
        // A positive subnormal p may be flushed to zero by a backend. Preserve
        // the first category explicitly; original validity is checked below.
        let first = logits
            .graph()
            .iota_i32(logits.shape(), axis)?
            .le_mask(&logits.graph().scalar_i32(0)?.broadcast_to(logits.shape())?)?;
        keep.maximum(&first)?.select(&sorted, &masked)?
    };
    let draw = categorical_from_bits(&filtered, bits, axis)?;
    let mut selected_shape = logits.shape().to_vec();
    selected_shape[axis] = 1;
    let indices = candidates
        .take_along_axis(&draw.indices.reshape(&selected_shape)?, axis)?
        .reshape(draw.indices.shape())?;
    let finite = logits.is_finite_mask()?;
    let valid = finite
        .add(&logits.eq_mask(&masked)?)?
        .min(&[axis], false)?
        .mul(&finite.max(&[axis], false)?)?
        .mul(&draw.valid)?;
    let indices = valid.select(
        &indices,
        &logits
            .graph()
            .scalar_i32(0)?
            .broadcast_to(indices.shape())?,
    )?;
    Ok(CategoricalSample { indices, valid })
}

/// Sample among the largest k logits, returning original category IDs.
/// `bits` has the logits shape with `axis` replaced by k; words are assigned to
/// candidates in descending stable top-k order, not original category order.
/// Requires 1 <= k <= axis size. Ties at the cutoff retain lower original IDs.
/// All original logits are validated, including discarded candidates: NaN,
/// +infinity, or no finite category yields valid=0 and placeholder ID zero.
/// Finite logits and -infinity masks are accepted. This applies the same
/// finite-precision Gumbel policy as `categorical_from_bits` to the truncated
/// distribution; k equal to the axis size need not reproduce that function's
/// draw because the random words are assigned in a different order.
/// No RNG state is committed and no temperature or top-p policy is inferred.
pub fn topk_categorical_from_bits(
    logits: &Tensor,
    bits: &Tensor,
    k: usize,
    axis: usize,
) -> Result<CategoricalSample> {
    let shape = topk_sample_shape(logits, k, axis)?;
    if !Arc::ptr_eq(&logits.graph().0, &bits.graph().0) || bits.shape() != shape {
        return Err(err(
            "top-k sampling bits must match candidate shape and graph",
        ));
    }
    let (values, candidates) = logits.topk(k, axis)?;
    let draw = categorical_from_bits(&values, bits, axis)?;
    let mut selected_shape = shape;
    selected_shape[axis] = 1;
    let indices = candidates
        .take_along_axis(&draw.indices.reshape(&selected_shape)?, axis)?
        .reshape(draw.indices.shape())?;
    let finite = logits.is_finite_mask()?;
    let masked = logits
        .graph()
        .constant(&[], &[f32::NEG_INFINITY])?
        .broadcast_to(logits.shape())?;
    let valid = finite
        .add(&logits.eq_mask(&masked)?)?
        .min(&[axis], false)?
        .mul(&finite.max(&[axis], false)?)?
        .mul(&draw.valid)?;
    let indices = valid.select(
        &indices,
        &logits
            .graph()
            .scalar_i32(0)?
            .broadcast_to(indices.shape())?,
    )?;
    Ok(CategoricalSample { indices, valid })
}

fn topk_sample_shape(logits: &Tensor, k: usize, axis: usize) -> Result<Vec<i64>> {
    if axis >= logits.shape().len()
        || logits.shape()[axis] > i32::MAX as i64
        || k == 0
        || k as u128 > logits.shape()[axis] as u128
    {
        return Err(err(
            "top-k sampling requires a valid I32-sized axis and 1 <= k <= axis size",
        ));
    }
    let mut shape = logits.shape().to_vec();
    shape[axis] = k as i64;
    Ok(shape)
}

/// Draw categorical indices with Gumbel-max using explicit random words.
/// `bits` must have the same graph and exact shape as `logits`. The category
/// axis must be nonempty. Finite logits and -infinity masks are accepted;
/// NaN, +infinity, or a row with no finite category invalidate the whole row.
/// No RNG state is advanced: supply independent uniform words (for example
/// from Threefry) and explicitly commit their proposed counter separately.
///
/// Uses a 23-bit midpoint grid strictly inside (0, 1), so this is a finite
/// precision approximation, not exact sampling of arbitrarily rare events.
/// Logits are centered before adding noise to preserve noise for large equal
/// logits. Floating-point ties use argmax's lowest-index rule; results near
/// ties need not agree across backends. Indices are nondifferentiable.
pub fn categorical_from_bits(
    logits: &Tensor,
    bits: &Tensor,
    axis: usize,
) -> Result<CategoricalSample> {
    if !Arc::ptr_eq(&logits.graph().0, &bits.graph().0) || logits.shape() != bits.shape() {
        return Err(err(
            "categorical logits/bits must have the same graph and shape",
        ));
    }
    if axis >= logits.shape().len()
        || logits.shape()[axis] == 0
        || logits.shape()[axis] > i32::MAX as i64
    {
        return Err(err("categorical axis must be valid and nonempty"));
    }
    let graph = &logits.graph();
    let masked = graph
        .constant(&[], &[f32::NEG_INFINITY])?
        .broadcast_to(logits.shape())?;
    let finite = logits.is_finite_mask()?;
    let allowed = finite.add(&logits.eq_mask(&masked)?)?;
    let valid = allowed
        .min(&[axis], false)?
        .mul(&finite.max(&[axis], false)?)?;
    // Sanitize invalid entries before reductions; the validity mask remains
    // authoritative even though an invalid row has a deterministic result.
    let safe = finite.select(logits, &masked)?;
    let maximum = safe.max(&[axis], true)?;
    let maximum = maximum.is_finite_mask()?.select(
        &maximum,
        &graph.constant(&[], &[0.])?.broadcast_to(maximum.shape())?,
    )?;
    let centered = safe.sub(&maximum.broadcast_to(logits.shape())?)?;
    let uniform = bits
        .shift_right_logical(9)?
        .to_f32()?
        .add_scalar(0.5)?
        .mul_scalar(1. / 8_388_608.)?;
    let noise = uniform.log()?.neg()?.log()?.neg()?;
    let scores = finite.select(&centered.add(&noise)?, &masked)?;
    let indices = scores.argmax(axis, false)?;
    let indices = valid.select(
        &indices,
        &graph.scalar_i32(0)?.broadcast_to(indices.shape())?,
    )?;
    Ok(CategoricalSample { indices, valid })
}

/// Pure batch draw: two random words per element and a proposed scalar counter.
/// No state is committed until the caller records its desired writes.
pub struct ThreefryBlocks {
    pub bits: [Tensor; 2],
    /// Low word then high word of the counter after reserving all blocks.
    pub next_counter: [Tensor; 2],
    /// Scalar F32 0/1: advancing the counter crossed the 64-bit period boundary.
    /// This is a warning value, not an automatic rejection or reseed operation.
    pub counter_wrapped: Tensor,
}

/// Allocate consecutive Threefry2x32 blocks from scalar key/counter words.
/// Counter order is low word, high word; element i uses counter+i in row-major
/// order. Returns two tensors of `shape`, consuming one 64-bit block per element
/// even if the caller uses only one output word. Scalar shape consumes one block;
/// empty shapes consume none. At most i32::MAX elements are supported per call.
///
/// Addition is modulo 2^64, with explicit low-word carry. `counter_wrapped`
/// reports wrap of the proposed end counter; callers must prevent unintended
/// reuse, including overlapping draws made from the same uncommitted counter.
/// Key/counter inputs must be scalar and belong to the same graph. This helper
/// only builds pure expressions: callers can commit next_counter jointly with
/// optimizer updates after computing a loss/acceptance predicate.
pub fn threefry2x32_blocks(
    key: [&Tensor; 2],
    counter: [&Tensor; 2],
    shape: &[i64],
) -> Result<ThreefryBlocks> {
    let count = elements(shape)?;
    if count > i32::MAX as usize {
        return Err(err("Threefry block count exceeds I32 iota range"));
    }
    for value in [key[0], key[1], counter[0], counter[1]] {
        if !value.shape().is_empty() || !Arc::ptr_eq(&key[0].graph().0, &value.graph().0) {
            return Err(err("Threefry batch key/counter must be same-graph scalars"));
        }
    }
    let graph = &key[0].graph();
    let offset = graph.iota_i32(&[count as i64], 0)?.reshape(shape)?;
    let low = counter[0].broadcast_to(shape)?;
    let high = counter[1].broadcast_to(shape)?;
    let (lanes, _) = add_counter(&low, &high, &offset)?;
    let keys = [key[0].broadcast_to(shape)?, key[1].broadcast_to(shape)?];
    let bits = threefry2x32([&keys[0], &keys[1]], [&lanes[0], &lanes[1]])?;
    let (next_counter, counter_wrapped) =
        add_counter(counter[0], counter[1], &graph.scalar_i32(count as i32)?)?;
    Ok(ThreefryBlocks {
        bits,
        next_counter,
        counter_wrapped,
    })
}

fn unsigned_less(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let sign = a.graph().scalar_i32(i32::MIN)?.broadcast_to(a.shape())?;
    b.bitwise_xor(&sign)?
        .le_mask(&a.bitwise_xor(&sign)?)?
        .neg()?
        .add_scalar(1.)
}

fn add_counter(low: &Tensor, high: &Tensor, increment: &Tensor) -> Result<([Tensor; 2], Tensor)> {
    let next_low = low.wrapping_add(increment)?;
    let carry = unsigned_less(&next_low, low)?;
    let next_high = carry.select(&high.wrapping_add_scalar(1)?, high)?;
    let wrapped = unsigned_less(&next_high, high)?;
    Ok(([next_low, next_high], wrapped))
}

/// Convert uniform 32-bit bit patterns to F32 uniform samples in [0, 1).
/// Uses the high 24 bits, exactly converts them to F32, then scales by 2^-24.
/// The output is a discrete grid: 0 is possible, 1 is not; the low 8 bits are
/// discarded. No entropy is added, state advanced, or distribution inferred
/// from arbitrary nonuniform inputs. Integer inputs are nondifferentiable.
/// Unlike converting all 32 bits to F32 before scaling, this cannot round to 1.
pub fn uniform_f32_from_bits(bits: &Tensor) -> Result<Tensor> {
    bits.shift_right_logical(8)?
        .to_f32()?
        .mul_scalar(1. / 16_777_216.)
}

/// Approximate standard-normal F32 samples using Box–Muller, one output per
/// pair of same-shape, same-graph I32 word tensors. Word0's high 23 bits select
/// a midpoint in (0, 1) for the radius; word1's high 24 bits select an angle in
/// [0, 2*pi). The sine companion is discarded. Uniform random words are required
/// for the distribution claim; arbitrary inputs add no entropy.
///
/// The finite uniform grid bounds the radius by sqrt(48*ln(2)) (about 5.77),
/// so this is not an exact unbounded Gaussian. Transcendental rounding can differ
/// across backends. No state is advanced; integer inputs are nondifferentiable.
pub fn normal_f32_from_bits(bits: [&Tensor; 2]) -> Result<Tensor> {
    if !Arc::ptr_eq(&bits[0].graph().0, &bits[1].graph().0) || bits[0].shape() != bits[1].shape() {
        return Err(err("normal bits must have the same shape and graph"));
    }
    let radius = bits[0]
        .shift_right_logical(9)?
        .to_f32()?
        .add_scalar(0.5)?
        .mul_scalar(1. / 8_388_608.)?
        .log()?
        .mul_scalar(-2.)?
        .sqrt()?;
    radius.mul(
        &uniform_f32_from_bits(bits[1])?
            .mul_scalar(std::f32::consts::TAU)?
            .cos()?,
    )
}

/// Threefry2x32 with 20 rounds, returning two I32 tensors of random bit patterns.
/// Key and counter each contain two words; all four tensors must have the same
/// graph and shape (broadcast scalars explicitly). Each tensor element is an
/// independent block invocation. Signed I32 represents the full u32 bit pattern.
///
/// This is a pure function: equal key/counter pairs reproduce the same block.
/// It does not advance state, assign streams, generate seed entropy, or promise
/// unique counters. The caller must prevent accidental key/counter reuse and
/// define batching/word order. No floating-point conversion or custom kernel
/// is involved. This is not a cryptographic API or a JAX seed/split compatibility
/// layer. Algorithm: Random123 Threefry2x32-20.
pub fn threefry2x32(key: [&Tensor; 2], counter: [&Tensor; 2]) -> Result<[Tensor; 2]> {
    let first = key[0];
    for value in [key[1], counter[0], counter[1]] {
        if !Arc::ptr_eq(&first.graph().0, &value.graph().0) || first.shape() != value.shape() {
            return Err(err("Threefry key/counter words must match graph and shape"));
        }
    }
    let parity = first
        .graph()
        .scalar_i32(0x1bd11bda)?
        .broadcast_to(first.shape())?;
    let third = key[0].bitwise_xor(key[1])?.bitwise_xor(&parity)?;
    let keys = [key[0], key[1], &third];
    let mut a = counter[0].wrapping_add(keys[0])?;
    let mut b = counter[1].wrapping_add(keys[1])?;
    let rotations = [13, 15, 26, 6, 17, 29, 16, 24];
    for round in 0..20 {
        let rotation = rotations[round % rotations.len()];
        a = a.wrapping_add(&b)?;
        b = b
            .shift_left(rotation)?
            .bitwise_or(&b.shift_right_logical(32 - rotation)?)?
            .bitwise_xor(&a)?;
        if (round + 1) % 4 == 0 {
            let injection = (round + 1) / 4;
            a = a.wrapping_add(keys[injection % 3])?;
            b = b
                .wrapping_add(keys[(injection + 1) % 3])?
                .wrapping_add_scalar(injection as i32)?;
        }
    }
    Ok([a, b])
}
