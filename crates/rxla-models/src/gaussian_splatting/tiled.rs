//! Portable tiled rasterization following the algorithmic structure of the
//! MIT-licensed JAX-GS renderer: https://github.com/ghif/jax-gs.

use super::*;
use rxla_core::DType;

/// Static budgets for the portable tiled StableHLO renderer.
#[derive(Clone, Copy, Debug)]
pub struct TiledGaussianConfig {
    pub tile_size: i64,
    pub max_tile_span: i64,
    pub max_per_tile: i64,
    pub chunk_size: i64,
}

impl Default for TiledGaussianConfig {
    fn default() -> Self {
        Self {
            tile_size: 16,
            max_tile_span: 8,
            max_per_tile: 4096,
            chunk_size: 512,
        }
    }
}

/// Tiled image plus explicit static-budget diagnostics.
pub struct TiledGaussianRender {
    pub color: Tensor,
    pub alpha: Tensor,
    pub tile_counts: Tensor,
    pub truncated_interactions: Tensor,
    pub span_overflow_gaussians: Tensor,
}

fn broadcast_vector(value: &Tensor, count: i64, interactions: i64) -> Result<Tensor> {
    Ok(value
        .reshape(&[count, 1])?
        .broadcast_to(&[count, interactions])?)
}

/// Render with static Gaussian/tile interactions and bounded per-tile work.
///
/// This is the portable counterpart of a CUDA tiled rasterizer. Every Gaussian
/// may cover at most `max_tile_span²` tiles and every tile composites at most
/// `max_per_tile` contributors. Both limits are explicit static-shape budgets.
pub fn render_gaussians_tiled(
    means: &Tensor,
    covariances: &Tensor,
    colors: &Tensor,
    opacities: &Tensor,
    background: &Tensor,
    camera: GaussianCamera,
    config: TiledGaussianConfig,
) -> Result<TiledGaussianRender> {
    let count = means.shape().first().copied().unwrap_or_default();
    let valid = count > 0
        && means.shape() == [count, 3]
        && covariances.shape() == [count, 3, 3]
        && colors.shape() == [count, 3]
        && opacities.shape() == [count]
        && background.shape() == [3]
        && means.same_trace(covariances)
        && means.same_trace(colors)
        && means.same_trace(opacities)
        && means.same_trace(background)
        && camera.width > 0
        && camera.height > 0
        && camera.focal_x.is_finite()
        && camera.focal_x > 0.0
        && camera.focal_y.is_finite()
        && camera.focal_y > 0.0
        && camera.near.is_finite()
        && camera.far.is_finite()
        && camera.near > 0.0
        && camera.far > camera.near
        && camera.covariance_epsilon.is_finite()
        && camera.covariance_epsilon > 0.0
        && camera.alpha_threshold.is_finite()
        && camera.alpha_threshold >= 0.0
        && camera.alpha_threshold < 0.99
        && config.tile_size > 0
        && config.max_tile_span > 0
        && config.max_per_tile > 0
        && config.chunk_size > 0
        && config.max_per_tile % config.chunk_size == 0;
    if !valid {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidGaussianSplattingInput,
        });
    }
    let tiles_x = (camera.width + config.tile_size - 1) / config.tile_size;
    let tiles_y = (camera.height + config.tile_size - 1) / config.tile_size;
    let tile_count = tiles_x.checked_mul(tiles_y).ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::InvalidGaussianSplattingInput,
    })?;
    let interactions = config
        .max_tile_span
        .checked_mul(config.max_tile_span)
        .ok_or(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidGaussianSplattingInput,
        })?;
    let flattened = count.checked_mul(interactions).ok_or(Error::InvalidModel {
        kind: ModelDefinitionError::InvalidGaussianSplattingInput,
    })?;
    if tile_count >= i32::MAX as i64 || flattened > i32::MAX as i64 {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidGaussianSplattingInput,
        });
    }

    let x = column(means, 0)?;
    let y = column(means, 1)?;
    let z = column(means, 2)?;
    let near = z.scalar(camera.near)?.broadcast_to(&[count])?;
    let far = z.scalar(camera.far)?.broadcast_to(&[count])?;
    let depth_valid = z.gt_mask(&near)?.mul(&z.lt_mask(&far)?)?.detach()?;
    let safe_z = depth_valid.select(&z, &near)?;
    let inv_z = z.scalar(1.0)?.broadcast_to(&[count])?.div(&safe_z)?;
    let inv_z2 = inv_z.square()?;
    let j00 = inv_z.mul_scalar(camera.focal_x)?;
    let j02 = x.mul(&inv_z2)?.mul_scalar(-camera.focal_x)?;
    let j11 = inv_z.mul_scalar(camera.focal_y)?;
    let j12 = y.mul(&inv_z2)?.mul_scalar(-camera.focal_y)?;
    let cxx = covariance(covariances, 0, 0)?;
    let cxy = covariance(covariances, 0, 1)?;
    let cxz = covariance(covariances, 0, 2)?;
    let cyy = covariance(covariances, 1, 1)?;
    let cyz = covariance(covariances, 1, 2)?;
    let czz = covariance(covariances, 2, 2)?;
    let var_x = j00
        .square()?
        .mul(&cxx)?
        .add(&j00.mul(&j02)?.mul(&cxz)?.mul_scalar(2.0)?)?
        .add(&j02.square()?.mul(&czz)?)?
        .add_scalar(camera.covariance_epsilon)?;
    let var_y = j11
        .square()?
        .mul(&cyy)?
        .add(&j11.mul(&j12)?.mul(&cyz)?.mul_scalar(2.0)?)?
        .add(&j12.square()?.mul(&czz)?)?
        .add_scalar(camera.covariance_epsilon)?;
    let cov_xy = j00
        .mul(&j11)?
        .mul(&cxy)?
        .add(&j00.mul(&j12)?.mul(&cxz)?)?
        .add(&j02.mul(&j11)?.mul(&cyz)?)?
        .add(&j02.mul(&j12)?.mul(&czz)?)?;
    let determinant = var_x
        .mul(&var_y)?
        .sub(&cov_xy.square()?)?
        .clamp(1e-12, f32::INFINITY)?;
    let inverse_xx = var_y.div(&determinant)?;
    let inverse_xy = cov_xy.neg()?.div(&determinant)?;
    let inverse_yy = var_x.div(&determinant)?;
    let projected_x = x
        .mul(&inv_z)?
        .mul_scalar(camera.focal_x)?
        .add_scalar(camera.center_x)?;
    let projected_y = y
        .mul(&inv_z)?
        .mul_scalar(camera.focal_y)?
        .add_scalar(camera.center_y)?;
    let trace = var_x.add(&var_y)?;
    let midpoint = trace.mul_scalar(0.5)?;
    let eigen = midpoint.add(
        &midpoint
            .square()?
            .sub(&determinant)?
            .maximum(&midpoint.scalar(0.0)?.broadcast_to(&[count])?)?
            .sqrt()?,
    )?;
    let radius = eigen.sqrt()?.mul_scalar(3.0)?.ceil()?;
    let on_screen = projected_x
        .add(&radius)?
        .gt_mask(&projected_x.scalar(0.0)?.broadcast_to(&[count])?)?
        .mul(
            &projected_x.sub(&radius)?.lt_mask(
                &projected_x
                    .scalar(camera.width as f32)?
                    .broadcast_to(&[count])?,
            )?,
        )?
        .mul(
            &projected_y
                .add(&radius)?
                .gt_mask(&projected_y.scalar(0.0)?.broadcast_to(&[count])?)?,
        )?
        .mul(
            &projected_y.sub(&radius)?.lt_mask(
                &projected_y
                    .scalar(camera.height as f32)?
                    .broadcast_to(&[count])?,
            )?,
        )?
        .mul(&depth_valid)?
        .detach()?;

    let tile_scale = 1.0 / config.tile_size as f32;
    let tile_min_x = projected_x
        .sub(&radius)?
        .mul_scalar(tile_scale)?
        .floor()?
        .clamp(0.0, (tiles_x - 1) as f32)?
        .cast(DType::I32)?;
    let tile_max_x = projected_x
        .add(&radius)?
        .mul_scalar(tile_scale)?
        .floor()?
        .clamp(0.0, (tiles_x - 1) as f32)?
        .cast(DType::I32)?;
    let tile_min_y = projected_y
        .sub(&radius)?
        .mul_scalar(tile_scale)?
        .floor()?
        .clamp(0.0, (tiles_y - 1) as f32)?
        .cast(DType::I32)?;
    let tile_max_y = projected_y
        .add(&radius)?
        .mul_scalar(tile_scale)?
        .floor()?
        .clamp(0.0, (tiles_y - 1) as f32)?
        .cast(DType::I32)?;
    let span_limit = means
        .scalar(config.max_tile_span as f32)?
        .cast(DType::I32)?
        .broadcast_to(&[count])?;
    let span_overflow_gaussians = tile_max_x
        .wrapping_sub(&tile_min_x)?
        .wrapping_add_scalar(1)?
        .gt_mask(&span_limit)?
        .maximum(
            &tile_max_y
                .wrapping_sub(&tile_min_y)?
                .wrapping_add_scalar(1)?
                .gt_mask(&span_limit)?,
        )?
        .mul(&on_screen)?
        .sum(&[0], false)?;

    let offset_index = means.iota_i32(&[1, interactions], 1)?;
    let offsets_y = offset_index
        .to_f32()?
        .mul_scalar(1.0 / config.max_tile_span as f32)?
        .floor()?
        .cast(DType::I32)?;
    let offsets_x = offset_index.wrapping_sub(
        &offsets_y.wrapping_mul(
            &means
                .scalar(config.max_tile_span as f32)?
                .cast(DType::I32)?
                .broadcast_to(&[1, interactions])?,
        )?,
    )?;
    let offsets_x = offsets_x.broadcast_to(&[count, interactions])?;
    let offsets_y = offsets_y.broadcast_to(&[count, interactions])?;
    let absolute_x = tile_min_x
        .reshape(&[count, 1])?
        .broadcast_to(&[count, interactions])?
        .wrapping_add(&offsets_x)?;
    let absolute_y = tile_min_y
        .reshape(&[count, 1])?
        .broadcast_to(&[count, interactions])?
        .wrapping_add(&offsets_y)?;
    let interaction_valid = absolute_x
        .le_mask(
            &tile_max_x
                .reshape(&[count, 1])?
                .broadcast_to(&[count, interactions])?,
        )?
        .mul(
            &absolute_y.le_mask(
                &tile_max_y
                    .reshape(&[count, 1])?
                    .broadcast_to(&[count, interactions])?,
            )?,
        )?
        .mul(&broadcast_vector(&on_screen, count, interactions)?)?;
    let tile_ids = absolute_y
        .wrapping_mul(
            &means
                .scalar(tiles_x as f32)?
                .cast(DType::I32)?
                .broadcast_to(&[count, interactions])?,
        )?
        .wrapping_add(&absolute_x)?;
    let sentinel = means
        .scalar(tile_count as f32)?
        .cast(DType::I32)?
        .broadcast_to(&[count, interactions])?;
    let tile_ids = interaction_valid
        .select(&tile_ids, &sentinel)?
        .reshape(&[flattened])?;
    let depths = interaction_valid
        .select(
            &broadcast_vector(&z, count, interactions)?,
            &z.scalar(f32::INFINITY)?
                .broadcast_to(&[count, interactions])?,
        )?
        .reshape(&[flattened])?;
    let gaussian_ids = means
        .iota_i32(&[count, interactions], 0)?
        .reshape(&[flattened])?;
    let depth_order = depths.argsort(0, false)?;
    let tile_ids = tile_ids.take_along_axis(&depth_order, 0)?;
    let gaussian_ids = gaussian_ids.take_along_axis(&depth_order, 0)?;
    let tile_order = tile_ids.to_f32()?.argsort(0, false)?;
    let sorted_tiles = tile_ids.take_along_axis(&tile_order, 0)?;
    let sorted_gaussians = gaussian_ids.take_along_axis(&tile_order, 0)?;

    let needles = means.iota_i32(&[tile_count + 1], 0)?;
    let boundaries = sorted_tiles.searchsorted_i32(&needles, false)?;
    let starts = boundaries.narrow(0, 0, tile_count)?;
    let ends = boundaries.narrow(0, 1, tile_count)?;
    let counts = ends.wrapping_sub(&starts)?;
    let tile_counts = counts.to_f32()?;
    let truncated_interactions = tile_counts
        .add_scalar(-(config.max_per_tile as f32))?
        .maximum(&tile_counts.scalar(0.0)?.broadcast_to(&[tile_count])?)?
        .sum(&[0], false)?;
    let pixels = config.tile_size * config.tile_size;
    let tile_index = means.iota_i32(&[tile_count, 1], 0)?.to_f32()?;
    let tile_y = tile_index.mul_scalar(1.0 / tiles_x as f32)?.floor()?;
    let tile_x = tile_index.sub(&tile_y.mul_scalar(tiles_x as f32)?)?;
    let pixel_index = means.iota_i32(&[1, pixels], 1)?.to_f32()?;
    let local_y = pixel_index
        .mul_scalar(1.0 / config.tile_size as f32)?
        .floor()?;
    let local_x = pixel_index.sub(&local_y.mul_scalar(config.tile_size as f32)?)?;
    let pixel_x = tile_x
        .mul_scalar(config.tile_size as f32)?
        .broadcast_to(&[tile_count, pixels])?
        .add(&local_x.broadcast_to(&[tile_count, pixels])?)?
        .add_scalar(0.5)?;
    let pixel_y = tile_y
        .mul_scalar(config.tile_size as f32)?
        .broadcast_to(&[tile_count, pixels])?
        .add(&local_y.broadcast_to(&[tile_count, pixels])?)?
        .add_scalar(0.5)?;
    let opacity = opacities.clamp(0.0, 0.99)?;
    let color = pixel_x
        .mul_scalar(0.0)?
        .unsqueeze(2)?
        .broadcast_to(&[tile_count, pixels, 3])?;
    let remaining = pixel_x.mul_scalar(0.0)?.add_scalar(1.0)?;
    let base_offsets = means.iota_i32(&[1, config.chunk_size], 1)?;
    let starts = starts.reshape(&[tile_count, 1])?;
    let counts = counts.reshape(&[tile_count, 1])?;
    let chunk_count = config.max_per_tile / config.chunk_size;
    let counter = means.scalar(0.0)?.cast(DType::I32)?;
    let composite_chunk =
        |carry: &[Tensor]| -> rxla_core::Result<Vec<Tensor>> {
            let color = &carry[1];
            let remaining = &carry[2];
            let chunk_base = carry[0]
                .wrapping_mul(&carry[0].scalar_i32(config.chunk_size as i32)?)?
                .broadcast_to(&[1, config.chunk_size])?;
            let offsets = base_offsets.wrapping_add(&chunk_base)?;
            let offsets = offsets.broadcast_to(&[tile_count, config.chunk_size])?;
            let contributor_indices = starts
                .broadcast_to(&[tile_count, config.chunk_size])?
                .wrapping_add(&offsets)?;
            let contributor_valid = offsets
                .lt_mask(&counts.broadcast_to(&[tile_count, config.chunk_size])?)?
                .detach()?;
            let ids = sorted_gaussians.take(&contributor_indices, 0)?;
            let gathered_x = projected_x.take(&ids, 0)?;
            let gathered_y = projected_y.take(&ids, 0)?;
            let dx = pixel_x
                .reshape(&[tile_count, pixels, 1])?
                .broadcast_to(&[tile_count, pixels, config.chunk_size])?
                .sub(
                    &gathered_x
                        .reshape(&[tile_count, 1, config.chunk_size])?
                        .broadcast_to(&[tile_count, pixels, config.chunk_size])?,
                )?;
            let dy = pixel_y
                .reshape(&[tile_count, pixels, 1])?
                .broadcast_to(&[tile_count, pixels, config.chunk_size])?
                .sub(
                    &gathered_y
                        .reshape(&[tile_count, 1, config.chunk_size])?
                        .broadcast_to(&[tile_count, pixels, config.chunk_size])?,
                )?;
            let expand = |value: &Tensor| {
                value
                    .take(&ids, 0)?
                    .reshape(&[tile_count, 1, config.chunk_size])?
                    .broadcast_to(&[tile_count, pixels, config.chunk_size])
            };
            let exponent = dx
                .square()?
                .mul(&expand(&inverse_xx)?)?
                .add(&dx.mul(&dy)?.mul(&expand(&inverse_xy)?)?.mul_scalar(2.0)?)?
                .add(&dy.square()?.mul(&expand(&inverse_yy)?)?)?
                .mul_scalar(-0.5)?;
            let alpha = exponent.exp()?.mul(&expand(&opacity)?)?;
            let valid = contributor_valid
                .reshape(&[tile_count, 1, config.chunk_size])?
                .broadcast_to(&[tile_count, pixels, config.chunk_size])?;
            let active = remaining
                .gt_mask(&remaining.scalar(1e-4)?.broadcast_to(remaining.shape())?)?
                .unsqueeze(2)?
                .broadcast_to(&[tile_count, pixels, config.chunk_size])?;
            let support = alpha
                .gt_mask(
                    &alpha
                        .scalar(camera.alpha_threshold)?
                        .broadcast_to(alpha.shape())?,
                )?
                .mul(&valid)?
                .mul(&active)?
                .detach()?;
            let alpha = alpha.mul(&support)?.clamp(0.0, 0.99)?;
            let one_minus_alpha = alpha.mul_scalar(-1.0)?.add_scalar(1.0)?;
            let local_transmittance = one_minus_alpha
                .narrow(2, 0, config.chunk_size - 1)?
                .pad(&[[0, 0], [0, 0], [1, 0]], 1.0)?
                .cumprod(2)?;
            let weights = alpha.mul(&local_transmittance)?.mul(
                &remaining
                    .unsqueeze(2)?
                    .broadcast_to(&[tile_count, pixels, config.chunk_size])?,
            )?;
            let gathered_colors = colors.take(&ids, 0)?;
            let contribution = weights
                .unsqueeze(3)?
                .broadcast_to(&[tile_count, pixels, config.chunk_size, 3])?
                .mul(
                    &gathered_colors
                        .reshape(&[tile_count, 1, config.chunk_size, 3])?
                        .broadcast_to(&[tile_count, pixels, config.chunk_size, 3])?,
                )?
                .sum(&[2], false)?;
            let color = color.add(&contribution)?;
            let chunk_remaining = one_minus_alpha
                .cumprod(2)?
                .narrow(2, config.chunk_size - 1, 1)?
                .reshape(&[tile_count, pixels])?;
            let remaining = remaining.mul(&chunk_remaining)?;
            Ok(vec![carry[0].wrapping_add_scalar(1)?, color, remaining])
        };
    let carry = if chunk_count == 1 {
        composite_chunk(&[counter, color, remaining])?
    } else {
        Tensor::while_loop(
            &[counter, color, remaining],
            |carry| {
                carry[0]
                    .lt_mask(&carry[0].scalar_i32(chunk_count as i32)?)?
                    .cast(DType::I32)
            },
            composite_chunk,
        )?
    };
    let mut color = carry[1].clone();
    let remaining = &carry[2];
    let accumulated = remaining.mul_scalar(-1.0)?.add_scalar(1.0)?;
    color = color.add(
        &remaining
            .unsqueeze(2)?
            .broadcast_to(&[tile_count, pixels, 3])?
            .mul(
                &background
                    .reshape(&[1, 1, 3])?
                    .broadcast_to(&[tile_count, pixels, 3])?,
            )?,
    )?;
    let padded_height = tiles_y * config.tile_size;
    let padded_width = tiles_x * config.tile_size;
    let color = color
        .reshape(&[tiles_y, tiles_x, config.tile_size, config.tile_size, 3])?
        .transpose(&[0, 2, 1, 3, 4])?
        .reshape(&[padded_height, padded_width, 3])?
        .slice(&[0, 0, 0], &[camera.height, camera.width, 3], &[1, 1, 1])?;
    let alpha = accumulated
        .reshape(&[tiles_y, tiles_x, config.tile_size, config.tile_size])?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[padded_height, padded_width])?
        .slice(&[0, 0], &[camera.height, camera.width], &[1, 1])?;
    Ok(TiledGaussianRender {
        color,
        alpha,
        tile_counts,
        truncated_interactions,
        span_overflow_gaussians,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client, Compiler, Tracer};

    fn scene(tracer: &Tracer) -> Result<TiledGaussianRender> {
        let means = tracer.constant(&[2, 3], &[0., 0., 2., 0., 0., 1.])?;
        let covariances = tracer.constant(
            &[2, 3, 3],
            &[
                0.04, 0., 0., 0., 0.04, 0., 0., 0., 0.04, 0.04, 0., 0., 0., 0.04, 0., 0., 0., 0.04,
            ],
        )?;
        let colors = tracer.constant(&[2, 3], &[0., 1., 0., 1., 0., 0.])?;
        let opacities = tracer.constant(&[2], &[0.5, 0.5])?;
        let background = tracer.constant(&[3], &[0., 0., 1.])?;
        render_gaussians_tiled(
            &means,
            &covariances,
            &colors,
            &opacities,
            &background,
            GaussianCamera::new(1, 1, 1., 1.).principal_point(0.5, 0.5),
            TiledGaussianConfig {
                tile_size: 1,
                max_tile_span: 1,
                max_per_tile: 2,
                chunk_size: 1,
            },
        )
    }

    #[test]
    fn tiled_renderer_lowers_to_static_stablehlo() {
        let tracer = Tracer::default();
        let render = scene(&tracer).unwrap();
        assert_eq!(render.color.shape(), [1, 1, 3]);
        assert_eq!(render.alpha.shape(), [1, 1]);
        let stablehlo = tracer
            .stablehlo_many(&[render.color, render.alpha])
            .unwrap();
        assert!(stablehlo.contains("stablehlo.sort"));
        assert!(stablehlo.contains("stablehlo.gather"));
        assert!(stablehlo.contains("stablehlo.select"));
        assert_eq!(stablehlo.matches("stablehlo.while").count(), 1);
        assert!(!stablehlo.contains("custom_call"));
    }

    #[test]
    fn tiled_renderer_keeps_continuous_gradients() {
        let tracer = Tracer::default();
        let means = tracer.input(&[1, 3]).unwrap();
        let covariance = tracer.input(&[1, 3, 3]).unwrap();
        let colors = tracer.input(&[1, 3]).unwrap();
        let opacity = tracer.input(&[1]).unwrap();
        let background = tracer.constant(&[3], &[0., 0., 0.]).unwrap();
        let render = render_gaussians_tiled(
            &means,
            &covariance,
            &colors,
            &opacity,
            &background,
            GaussianCamera::new(1, 1, 1., 1.).principal_point(0.5, 0.5),
            TiledGaussianConfig {
                tile_size: 1,
                max_tile_span: 1,
                max_per_tile: 1,
                chunk_size: 1,
            },
        )
        .unwrap();
        let loss = render.color.sum(&[0, 1, 2], false).unwrap();
        let gradients = loss.grad(&[means, covariance, colors, opacity]).unwrap();
        assert_eq!(gradients.len(), 4);
        let stablehlo = tracer.stablehlo_many(&gradients).unwrap();
        assert!(stablehlo.contains("stablehlo.gather"));
        assert!(!stablehlo.contains("custom_call"));
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn tiled_renderer_executes_front_to_back() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
        let tracer = Tracer::default();
        let render = scene(&tracer)?;
        let mut compiler = Compiler::new(client, CacheLimits::default());
        let executable = compiler.compile_many(&tracer, &[render.color, render.alpha])?;
        let outputs = executable.execute(&[])?;
        assert_eq!(outputs[0].to_vec::<f32>()?, [0.5, 0.25, 0.25]);
        assert_eq!(outputs[1].to_vec::<f32>()?, [0.75]);
        Ok(())
    }
}
