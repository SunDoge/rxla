//! Pure StableHLO reference implementation of 3D Gaussian splatting.
//!
//! This deliberately uses dense `[height, width, gaussians]` intermediates.
//! It is a correctness and compiler experiment, not a replacement for a tiled
//! CUDA rasterizer.

use crate::{Error, ModelDefinitionError, Result};
use rxla_core::Tensor;

/// Pinhole camera and rasterization constants.
#[derive(Clone, Copy, Debug)]
pub struct GaussianCamera {
    width: i64,
    height: i64,
    focal_x: f32,
    focal_y: f32,
    center_x: f32,
    center_y: f32,
    near: f32,
    far: f32,
    covariance_epsilon: f32,
}

impl GaussianCamera {
    pub fn new(width: i64, height: i64, focal_x: f32, focal_y: f32) -> Self {
        Self {
            width,
            height,
            focal_x,
            focal_y,
            center_x: width as f32 * 0.5,
            center_y: height as f32 * 0.5,
            near: 1e-2,
            far: 1e4,
            covariance_epsilon: 0.3,
        }
    }

    pub fn principal_point(mut self, x: f32, y: f32) -> Self {
        self.center_x = x;
        self.center_y = y;
        self
    }

    pub fn depth_range(mut self, near: f32, far: f32) -> Self {
        self.near = near;
        self.far = far;
        self
    }

    pub fn covariance_epsilon(mut self, epsilon: f32) -> Self {
        self.covariance_epsilon = epsilon;
        self
    }
}

/// Dense rendered RGB image and accumulated alpha.
pub struct GaussianRender {
    pub color: Tensor,
    pub alpha: Tensor,
}

fn column(input: &Tensor, index: i64) -> Result<Tensor> {
    Ok(input.narrow(1, index, 1)?.reshape(&[input.shape()[0]])?)
}

fn covariance(input: &Tensor, row: i64, column: i64) -> Result<Tensor> {
    Ok(input
        .narrow(1, row, 1)?
        .narrow(2, column, 1)?
        .reshape(&[input.shape()[0]])?)
}

/// Render camera-space 3D Gaussians using only ordinary RXLA/StableHLO ops.
///
/// - `means`: `[G, 3]`, in camera coordinates.
/// - `covariances`: `[G, 3, 3]`, symmetric positive covariance matrices.
/// - `colors`: `[G, 3]`, linear RGB.
/// - `opacities`: `[G]`, values are clamped to `[0, 0.99]`.
/// - `pixels`: `[height, width, 2]`, pixel-center x/y coordinates.
/// - `background`: `[3]`, linear RGB.
///
/// Gaussians are stably sorted front-to-back by camera-space depth. Sorting is
/// nondifferentiable, while selected means, covariance, opacity and color retain
/// their ordinary gradients. Gaussians outside the depth range contribute zero.
pub fn render_gaussians(
    means: &Tensor,
    covariances: &Tensor,
    colors: &Tensor,
    opacities: &Tensor,
    pixels: &Tensor,
    background: &Tensor,
    camera: GaussianCamera,
) -> Result<GaussianRender> {
    let count = means.shape().first().copied().unwrap_or_default();
    let valid = count > 0
        && means.shape() == [count, 3]
        && covariances.shape() == [count, 3, 3]
        && colors.shape() == [count, 3]
        && opacities.shape() == [count]
        && pixels.shape() == [camera.height, camera.width, 2]
        && background.shape() == [3]
        && means.same_trace(covariances)
        && means.same_trace(colors)
        && means.same_trace(opacities)
        && means.same_trace(pixels)
        && means.same_trace(background)
        && camera.width > 0
        && camera.height > 0
        && camera.focal_x.is_finite()
        && camera.focal_x > 0.0
        && camera.focal_y.is_finite()
        && camera.focal_y > 0.0
        && camera.center_x.is_finite()
        && camera.center_y.is_finite()
        && camera.near.is_finite()
        && camera.far.is_finite()
        && camera.near > 0.0
        && camera.far > camera.near
        && camera.covariance_epsilon.is_finite()
        && camera.covariance_epsilon > 0.0;
    if !valid {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidGaussianSplattingInput,
        });
    }

    let x = column(means, 0)?;
    let y = column(means, 1)?;
    let z = column(means, 2)?;
    let order = z.argsort(0, false)?;
    let x = x.take_along_axis(&order, 0)?;
    let y = y.take_along_axis(&order, 0)?;
    let z = z.take_along_axis(&order, 0)?;
    let sorted_covariance = covariances.take(&order, 0)?;
    let colors = colors.take(&order, 0)?;
    let opacity = opacities.take_along_axis(&order, 0)?.clamp(0.0, 0.99)?;

    // Keep rejected Gaussians numerically well-defined. Multiplying an invalid
    // projection by a zero visibility mask would not remove an earlier NaN.
    let near = z.scalar(camera.near)?.broadcast_to(&[count])?;
    let far = z.scalar(camera.far)?.broadcast_to(&[count])?;
    let depth_mask = z.gt_mask(&near)?.mul(&z.lt_mask(&far)?)?.detach()?;
    let safe_z = depth_mask.select(&z, &near)?;
    let inv_z = z.scalar(1.0)?.broadcast_to(&[count])?.div(&safe_z)?;
    let inv_z2 = inv_z.square()?;
    let j00 = inv_z.mul_scalar(camera.focal_x)?;
    let j02 = x.mul(&inv_z2)?.mul_scalar(-camera.focal_x)?;
    let j11 = inv_z.mul_scalar(camera.focal_y)?;
    let j12 = y.mul(&inv_z2)?.mul_scalar(-camera.focal_y)?;

    let cxx = covariance(&sorted_covariance, 0, 0)?;
    let cxy = covariance(&sorted_covariance, 0, 1)?;
    let cxz = covariance(&sorted_covariance, 0, 2)?;
    let cyy = covariance(&sorted_covariance, 1, 1)?;
    let cyz = covariance(&sorted_covariance, 1, 2)?;
    let czz = covariance(&sorted_covariance, 2, 2)?;

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
    let pixel_x = pixels
        .narrow(2, 0, 1)?
        .broadcast_to(&[camera.height, camera.width, count])?;
    let pixel_y = pixels
        .narrow(2, 1, 1)?
        .broadcast_to(&[camera.height, camera.width, count])?;
    let dx = pixel_x.sub(&projected_x.reshape(&[1, 1, count])?.broadcast_to(&[
        camera.height,
        camera.width,
        count,
    ])?)?;
    let dy = pixel_y.sub(&projected_y.reshape(&[1, 1, count])?.broadcast_to(&[
        camera.height,
        camera.width,
        count,
    ])?)?;
    let broadcast = |value: &Tensor| {
        value
            .reshape(&[1, 1, count])?
            .broadcast_to(&[camera.height, camera.width, count])
    };
    let exponent = dx
        .square()?
        .mul(&broadcast(&inverse_xx)?)?
        .add(
            &dx.mul(&dy)?
                .mul(&broadcast(&inverse_xy)?)?
                .mul_scalar(2.0)?,
        )?
        .add(&dy.square()?.mul(&broadcast(&inverse_yy)?)?)?
        .mul_scalar(-0.5)?;
    let alpha = exponent
        .exp()?
        .mul(&broadcast(&opacity)?)?
        .mul(&broadcast(&depth_mask)?)?
        .clamp(0.0, 0.99)?;

    let one_minus_alpha = alpha.mul_scalar(-1.0)?.add_scalar(1.0)?;
    let exclusive = one_minus_alpha
        .narrow(2, 0, count - 1)?
        .pad(&[[0, 0], [0, 0], [1, 0]], 1.0)?;
    let transmittance = exclusive.cumprod(2)?;
    let weights = alpha.mul(&transmittance)?;
    let accumulated = weights.sum(&[2], false)?;
    let color = weights
        .unsqueeze(3)?
        .broadcast_to(&[camera.height, camera.width, count, 3])?
        .mul(&colors.reshape(&[1, 1, count, 3])?.broadcast_to(&[
            camera.height,
            camera.width,
            count,
            3,
        ])?)?
        .sum(&[2], false)?;
    let remaining = accumulated
        .mul_scalar(-1.0)?
        .add_scalar(1.0)?
        .unsqueeze(2)?
        .broadcast_to(&[camera.height, camera.width, 3])?;
    let color = color.add(
        &remaining.mul(&background.reshape(&[1, 1, 3])?.broadcast_to(&[
            camera.height,
            camera.width,
            3,
        ])?)?,
    )?;
    Ok(GaussianRender {
        color,
        alpha: accumulated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client, Compiler, Tracer};

    fn two_gaussians(tracer: &Tracer) -> Result<GaussianRender> {
        // Deliberately store the rear green Gaussian first. Correct
        // front-to-back sorting yields red=0.5, green=0.25 at the center.
        let means = tracer.constant(&[2, 3], &[0., 0., 2., 0., 0., 1.])?;
        let covariances = tracer.constant(
            &[2, 3, 3],
            &[
                0.04, 0., 0., 0., 0.04, 0., 0., 0., 0.04, // rear
                0.04, 0., 0., 0., 0.04, 0., 0., 0., 0.04, // front
            ],
        )?;
        let colors = tracer.constant(&[2, 3], &[0., 1., 0., 1., 0., 0.])?;
        let opacities = tracer.constant(&[2], &[0.5, 0.5])?;
        let pixels = tracer.constant(&[1, 1, 2], &[0., 0.])?;
        let background = tracer.constant(&[3], &[0., 0., 1.])?;
        render_gaussians(
            &means,
            &covariances,
            &colors,
            &opacities,
            &pixels,
            &background,
            GaussianCamera::new(1, 1, 1., 1.).principal_point(0., 0.),
        )
    }

    #[test]
    fn renderer_is_pure_stablehlo_and_keeps_depth_sort() {
        let tracer = Tracer::default();
        let render = two_gaussians(&tracer).unwrap();
        assert_eq!(render.color.shape(), [1, 1, 3]);
        assert_eq!(render.alpha.shape(), [1, 1]);
        let stablehlo = tracer
            .stablehlo_many(&[render.color, render.alpha])
            .unwrap();
        assert!(stablehlo.contains("stablehlo.sort"));
        assert!(stablehlo.contains("stablehlo.exponential"));
        assert!(!stablehlo.contains("custom_call"));
    }

    #[test]
    fn renderer_gradients_stay_in_stablehlo() {
        let tracer = Tracer::default();
        let means = tracer.input(&[1, 3]).unwrap();
        let covariances = tracer.input(&[1, 3, 3]).unwrap();
        let colors = tracer.input(&[1, 3]).unwrap();
        let opacities = tracer.input(&[1]).unwrap();
        let pixels = tracer.constant(&[1, 1, 2], &[0.2, -0.1]).unwrap();
        let background = tracer.constant(&[3], &[0., 0., 0.]).unwrap();
        let render = render_gaussians(
            &means,
            &covariances,
            &colors,
            &opacities,
            &pixels,
            &background,
            GaussianCamera::new(1, 1, 1., 1.).principal_point(0., 0.),
        )
        .unwrap();
        let loss = render.color.sum(&[0, 1, 2], false).unwrap();
        let gradients = loss
            .grad(&[
                means.clone(),
                covariances.clone(),
                colors.clone(),
                opacities.clone(),
            ])
            .unwrap();
        assert_eq!(gradients[0].shape(), means.shape());
        assert_eq!(gradients[1].shape(), covariances.shape());
        assert_eq!(gradients[2].shape(), colors.shape());
        assert_eq!(gradients[3].shape(), opacities.shape());
        let stablehlo = tracer.stablehlo_many(&gradients).unwrap();
        assert!(stablehlo.contains("stablehlo.sort"));
        assert!(!stablehlo.contains("custom_call"));
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn renderer_executes_front_to_back_compositing()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
        let tracer = Tracer::default();
        let render = two_gaussians(&tracer)?;
        let mut compiler = Compiler::new(client, CacheLimits::default());
        let executable = compiler.compile_many(&tracer, &[render.color, render.alpha])?;
        let outputs = executable.execute(&[])?;
        assert_eq!(outputs[0].to_vec::<f32>()?, [0.5, 0.25, 0.25]);
        assert_eq!(outputs[1].to_vec::<f32>()?, [0.75]);
        Ok(())
    }

    #[test]
    #[ignore = "requires trusted PJRT_PLUGIN_PATH"]
    fn renderer_executes_gradients() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?) }?;
        let tracer = Tracer::default();
        let means = tracer.input(&[1, 3])?;
        let covariances = tracer.input(&[1, 3, 3])?;
        let colors = tracer.input(&[1, 3])?;
        let opacities = tracer.input(&[1])?;
        let pixels = tracer.constant(&[1, 1, 2], &[0.2, -0.1])?;
        let background = tracer.constant(&[3], &[0., 0., 0.])?;
        let render = render_gaussians(
            &means,
            &covariances,
            &colors,
            &opacities,
            &pixels,
            &background,
            GaussianCamera::new(1, 1, 1., 1.).principal_point(0., 0.),
        )?;
        let loss = render.color.sum(&[0, 1, 2], false)?;
        let gradients = loss.grad(&[
            means.clone(),
            covariances.clone(),
            colors.clone(),
            opacities.clone(),
        ])?;
        let mut compiler = Compiler::new(client, CacheLimits::default());
        let executable = compiler.compile_many(&tracer, &gradients)?;
        let means = compiler.client().buffer(&[1, 3], &[0.1, -0.1, 2.0])?;
        let covariances = compiler
            .client()
            .buffer(&[1, 3, 3], &[0.04, 0., 0., 0., 0.04, 0., 0., 0., 0.04])?;
        let colors = compiler.client().buffer(&[1, 3], &[0.8, 0.3, 0.1])?;
        let opacities = compiler.client().buffer(&[1], &[0.6])?;
        let outputs = executable.execute(&[&means, &covariances, &colors, &opacities])?;
        for output in &outputs {
            assert!(
                output
                    .to_vec::<f32>()?
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        assert!(
            outputs[2]
                .to_vec::<f32>()?
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            outputs[3]
                .to_vec::<f32>()?
                .iter()
                .any(|value| *value != 0.0)
        );
        Ok(())
    }
}
