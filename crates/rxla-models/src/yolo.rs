//! YOLOX detection models following the official Megvii module tree.

use super::*;
use derive_setters::Setters;
use rxla_core::{Conv2dOptions, Pool2dOptions, Tensor, vision};
use rxla_nn::{BatchNorm, Conv2d, Layer, TensorApply, path};
use snafu::{Snafu, ensure};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YoloConfig {
    classes: i64,
    depth: f32,
    width: f32,
    depthwise: bool,
}

impl YoloConfig {
    pub fn yolox_tiny(classes: i64) -> Self {
        Self {
            classes,
            depth: 0.33,
            width: 0.375,
            depthwise: true,
        }
    }

    pub fn yolox_s(classes: i64) -> Self {
        Self {
            classes,
            depth: 0.33,
            width: 0.5,
            depthwise: false,
        }
    }

    pub fn classes(self) -> i64 {
        self.classes
    }
    fn channels(self, base: i64) -> i64 {
        (base as f32 * self.width) as i64
    }
    fn backbone_depth(self) -> usize {
        (self.depth * 3.).round().max(1.) as usize
    }
    fn neck_depth(self) -> usize {
        (self.depth * 3.).round() as usize
    }

    fn validate(self) -> Result<()> {
        if self.classes <= 0
            || !self.depth.is_finite()
            || self.depth <= 0.
            || !self.width.is_finite()
            || self.width <= 0.
            || self.channels(64) <= 0
        {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidYoloConfiguration,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
struct BaseConv {
    channels: i64,
    kernel: i64,
    #[setters(generate)]
    stride: i64,
    #[setters(generate)]
    groups: i64,
}

impl BaseConv {
    fn new(channels: i64, kernel: i64) -> Self {
        Self {
            channels,
            kernel,
            stride: 1,
            groups: 1,
        }
    }
}

impl Layer for BaseConv {
    fn apply_in(&self, cx: &Cx, input: &Tensor) -> rxla_nn::Result<Tensor> {
        let hidden = input.through(
            path!(cx / "conv")?,
            Conv2d::new(self.channels, [self.kernel, self.kernel])
                .options(Conv2dOptions {
                    strides: [self.stride; 2],
                    padding: [[self.kernel / 2, self.kernel / 2]; 2],
                    groups: self.groups,
                    ..Default::default()
                })
                .bias(false),
        )?;
        hidden
            .through(path!(cx / "bn")?, BatchNorm::new())?
            .silu()
            .map_err(Into::into)
    }
}

#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
struct YoloConv {
    channels: i64,
    kernel: i64,
    #[setters(generate)]
    stride: i64,
    #[setters(generate)]
    depthwise: bool,
}

impl YoloConv {
    fn new(channels: i64, kernel: i64) -> Self {
        Self {
            channels,
            kernel,
            stride: 1,
            depthwise: false,
        }
    }
}

impl Layer for YoloConv {
    fn apply_in(&self, cx: &Cx, input: &Tensor) -> rxla_nn::Result<Tensor> {
        if !self.depthwise {
            return BaseConv::new(self.channels, self.kernel)
                .stride(self.stride)
                .apply_in(cx, input);
        }
        let channels = input.shape()[3];
        let hidden = input.through(
            path!(cx / "dconv")?,
            BaseConv::new(channels, self.kernel)
                .stride(self.stride)
                .groups(channels),
        )?;
        hidden.through(path!(cx / "pconv")?, BaseConv::new(self.channels, 1))
    }
}

#[derive(Clone, Copy, Debug)]
struct Focus {
    channels: i64,
}

impl Focus {
    fn new(channels: i64) -> Self {
        Self { channels }
    }
}

impl Layer for Focus {
    fn apply_in(&self, cx: &Cx, input: &Tensor) -> rxla_nn::Result<Tensor> {
        let [b, h, w, c] = input.shape() else {
            unreachable!("validated rank")
        };
        let limits = [*b, *h, *w, *c];
        let strides = [1, 2, 2, 1];
        let patches = [
            input.slice(&[0, 0, 0, 0], &limits, &strides)?,
            input.slice(&[0, 1, 0, 0], &limits, &strides)?,
            input.slice(&[0, 0, 1, 0], &limits, &strides)?,
            input.slice(&[0, 1, 1, 0], &limits, &strides)?,
        ];
        Tensor::concatenate(&patches, 3)?
            .through(path!(cx / "conv")?, YoloConv::new(self.channels, 3))
    }
}

#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
struct Bottleneck {
    channels: i64,
    #[setters(generate)]
    shortcut: bool,
    #[setters(generate)]
    depthwise: bool,
}

impl Bottleneck {
    fn new(channels: i64) -> Self {
        Self {
            channels,
            shortcut: true,
            depthwise: false,
        }
    }
}

impl Layer for Bottleneck {
    fn apply_in(&self, cx: &Cx, input: &Tensor) -> rxla_nn::Result<Tensor> {
        let hidden = input.through(path!(cx / "conv1")?, YoloConv::new(self.channels, 1))?;
        let output = hidden.through(
            path!(cx / "conv2")?,
            YoloConv::new(self.channels, 3).depthwise(self.depthwise),
        )?;
        if self.shortcut && input.shape() == output.shape() {
            input.add(&output).map_err(Into::into)
        } else {
            Ok(output)
        }
    }
}

#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
struct CspLayer {
    channels: i64,
    #[setters(generate)]
    repeats: usize,
    #[setters(generate)]
    shortcut: bool,
    #[setters(generate)]
    depthwise: bool,
}

impl CspLayer {
    fn new(channels: i64) -> Self {
        Self {
            channels,
            repeats: 1,
            shortcut: true,
            depthwise: false,
        }
    }
}

impl Layer for CspLayer {
    fn apply_in(&self, cx: &Cx, input: &Tensor) -> rxla_nn::Result<Tensor> {
        let hidden = self.channels / 2;
        let mut branch = input.through(path!(cx / "conv1")?, YoloConv::new(hidden, 1))?;
        let bypass = input.through(path!(cx / "conv2")?, YoloConv::new(hidden, 1))?;
        for index in 0..self.repeats {
            branch = branch.through(
                path!(cx / "m" / index)?,
                Bottleneck::new(hidden)
                    .shortcut(self.shortcut)
                    .depthwise(self.depthwise),
            )?;
        }
        Tensor::concatenate(&[branch, bypass], 3)?
            .through(path!(cx / "conv3")?, YoloConv::new(self.channels, 1))
    }
}

#[derive(Clone, Copy, Debug)]
struct Spp {
    channels: i64,
}

impl Spp {
    fn new(channels: i64) -> Self {
        Self { channels }
    }
}

impl Layer for Spp {
    fn apply_in(&self, cx: &Cx, input: &Tensor) -> rxla_nn::Result<Tensor> {
        let hidden = input.through(path!(cx / "conv1")?, YoloConv::new(self.channels / 2, 1))?;
        let mut values = vec![hidden.clone()];
        for window in [5, 9, 13] {
            values.push(hidden.max_pool2d(Pool2dOptions {
                window: [window; 2],
                strides: [1; 2],
                padding: [[window / 2, window / 2]; 2],
            })?);
        }
        Tensor::concatenate(&values, 3)?
            .through(path!(cx / "conv2")?, YoloConv::new(self.channels, 1))
    }
}

fn csp_darknet(cx: Cx, images: &Tensor, config: YoloConfig) -> Result<[Tensor; 3]> {
    let base = config.channels(64);
    let depth = config.backbone_depth();
    let stem = images.through(path!(cx / "stem")?, Focus::new(base))?;
    let dark2 = stem.through(
        path!(cx / "dark2" / 0)?,
        YoloConv::new(base * 2, 3)
            .stride(2)
            .depthwise(config.depthwise),
    )?;
    let dark2 = dark2.through(
        path!(cx / "dark2" / 1)?,
        CspLayer::new(base * 2)
            .repeats(depth)
            .depthwise(config.depthwise),
    )?;
    let dark3 = dark2.through(
        path!(cx / "dark3" / 0)?,
        YoloConv::new(base * 4, 3)
            .stride(2)
            .depthwise(config.depthwise),
    )?;
    let dark3 = dark3.through(
        path!(cx / "dark3" / 1)?,
        CspLayer::new(base * 4)
            .repeats(depth * 3)
            .depthwise(config.depthwise),
    )?;
    let dark4 = dark3.through(
        path!(cx / "dark4" / 0)?,
        YoloConv::new(base * 8, 3)
            .stride(2)
            .depthwise(config.depthwise),
    )?;
    let dark4 = dark4.through(
        path!(cx / "dark4" / 1)?,
        CspLayer::new(base * 8)
            .repeats(depth * 3)
            .depthwise(config.depthwise),
    )?;
    let dark5 = dark4.through(
        path!(cx / "dark5" / 0)?,
        YoloConv::new(base * 16, 3)
            .stride(2)
            .depthwise(config.depthwise),
    )?;
    let dark5 = dark5.through(path!(cx / "dark5" / 1)?, Spp::new(base * 16))?;
    let dark5 = dark5.through(
        path!(cx / "dark5" / 2)?,
        CspLayer::new(base * 16)
            .repeats(depth)
            .shortcut(false)
            .depthwise(config.depthwise),
    )?;
    Ok([dark3, dark4, dark5])
}

fn pafpn(cx: Cx, images: &Tensor, config: YoloConfig) -> Result<[Tensor; 3]> {
    let [x2, x1, x0] = csp_darknet(path!(cx / "backbone")?, images, config)?;
    let depth = config.neck_depth();
    let (c3, c4, c5) = (
        config.channels(256),
        config.channels(512),
        config.channels(1024),
    );
    let fpn0 = x0.through(path!(cx / "lateral_conv0")?, YoloConv::new(c4, 1))?;
    let top4 = Tensor::concatenate(&[fpn0.upsample_nearest2d([2, 2])?, x1], 3)?;
    let top4 = top4.through(
        path!(cx / "C3_p4")?,
        CspLayer::new(c4)
            .repeats(depth)
            .shortcut(false)
            .depthwise(config.depthwise),
    )?;
    let fpn1 = top4.through(path!(cx / "reduce_conv1")?, YoloConv::new(c3, 1))?;
    let top3 = Tensor::concatenate(&[fpn1.upsample_nearest2d([2, 2])?, x2], 3)?;
    let out2 = top3.through(
        path!(cx / "C3_p3")?,
        CspLayer::new(c3)
            .repeats(depth)
            .shortcut(false)
            .depthwise(config.depthwise),
    )?;
    let down2 = out2.through(
        path!(cx / "bu_conv2")?,
        YoloConv::new(c3, 3).stride(2).depthwise(config.depthwise),
    )?;
    let out1 = Tensor::concatenate(&[down2, fpn1], 3)?.through(
        path!(cx / "C3_n3")?,
        CspLayer::new(c4)
            .repeats(depth)
            .shortcut(false)
            .depthwise(config.depthwise),
    )?;
    let down1 = out1.through(
        path!(cx / "bu_conv1")?,
        YoloConv::new(c4, 3).stride(2).depthwise(config.depthwise),
    )?;
    let out0 = Tensor::concatenate(&[down1, fpn0], 3)?.through(
        path!(cx / "C3_n4")?,
        CspLayer::new(c5)
            .repeats(depth)
            .shortcut(false)
            .depthwise(config.depthwise),
    )?;
    Ok([out2, out1, out0])
}

fn head(cx: Cx, features: [Tensor; 3], config: YoloConfig) -> Result<[Tensor; 3]> {
    let width = config.channels(256);
    let mut outputs = Vec::with_capacity(3);
    for (index, feature) in features.into_iter().enumerate() {
        let stem = feature.through(path!(cx / "stems" / index)?, YoloConv::new(width, 1))?;
        let (mut cls, mut reg) = (stem.clone(), stem);
        for layer in 0..2 {
            cls = cls.through(
                path!(cx / "cls_convs" / index / layer)?,
                YoloConv::new(width, 3).depthwise(config.depthwise),
            )?;
            reg = reg.through(
                path!(cx / "reg_convs" / index / layer)?,
                YoloConv::new(width, 3).depthwise(config.depthwise),
            )?;
        }
        let boxes = reg.through(path!(cx / "reg_preds" / index)?, Conv2d::new(4, [1, 1]))?;
        let objectness = reg.through(path!(cx / "obj_preds" / index)?, Conv2d::new(1, [1, 1]))?;
        let classes = cls.through(
            path!(cx / "cls_preds" / index)?,
            Conv2d::new(config.classes, [1, 1]),
        )?;
        outputs.push(Tensor::concatenate(&[boxes, objectness, classes], 3)?);
    }
    debug_assert_eq!(outputs.len(), 3);
    Ok([outputs.remove(0), outputs.remove(0), outputs.remove(0)])
}

/// Official YOLOX topology with raw NHWC heads at strides 8, 16 and 32.
pub fn yolo(cx: Cx, images: &Tensor, config: &YoloConfig) -> Result<[Tensor; 3]> {
    config.validate()?;
    let [_, height, width, channels] = images.shape() else {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidYoloInput,
        });
    };
    if images.dtype() != DType::F32
        || *height <= 0
        || *width <= 0
        || *channels != 3
        || height % 32 != 0
        || width % 32 != 0
    {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidYoloInput,
        });
    }
    let features = pafpn(path!(cx / "backbone")?, images, *config)?;
    head(path!(cx / "head")?, features, *config)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub xyxy: [f32; 4],
    pub score: f32,
    pub class: i32,
}

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum YoloDecodeError {
    #[snafu(display("invalid YOLOX output shape or data length"))]
    Shape,
    #[snafu(display("invalid YOLOX confidence threshold"))]
    Confidence,
    #[snafu(transparent)]
    Nms { source: vision::NmsError },
}

/// Postprocessing policy for raw YOLOX heads.
#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
pub struct YoloDecodeConfig {
    #[setters(generate)]
    confidence: f32,
    #[setters(generate)]
    iou_threshold: f32,
    #[setters(generate)]
    max_detections: usize,
}

impl Default for YoloDecodeConfig {
    fn default() -> Self {
        Self {
            confidence: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
        }
    }
}

impl YoloDecodeConfig {
    /// Decode raw YOLOX heads into input-pixel coordinates, then class-aware NMS.
    pub fn decode(
        self,
        outputs: [(&[f32], [usize; 4], usize); 3],
    ) -> std::result::Result<Vec<Detection>, YoloDecodeError> {
        ensure!(
            self.confidence.is_finite() && (0. ..=1.).contains(&self.confidence),
            ConfidenceSnafu
        );
        let mut detections = Vec::new();
        for (values, [batch, height, width, channels], stride) in outputs {
            ensure!(
                batch == 1
                    && channels >= 6
                    && stride > 0
                    && values.len() == batch * height * width * channels,
                ShapeSnafu
            );
            for y in 0..height {
                for x in 0..width {
                    let row = &values[(y * width + x) * channels..][..channels];
                    let (cx, cy) = (
                        (row[0] + x as f32) * stride as f32,
                        (row[1] + y as f32) * stride as f32,
                    );
                    let (w, h) = (row[2].exp() * stride as f32, row[3].exp() * stride as f32);
                    let objectness = sigmoid(row[4]);
                    for (class, &logit) in row[5..].iter().enumerate() {
                        let score = objectness * sigmoid(logit);
                        if score >= self.confidence {
                            detections.push(Detection {
                                xyxy: [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
                                score,
                                class: class as i32,
                            });
                        }
                    }
                }
            }
        }
        let boxes = detections.iter().map(|v| v.xyxy).collect::<Vec<_>>();
        let scores = detections.iter().map(|v| v.score).collect::<Vec<_>>();
        let classes = detections.iter().map(|v| v.class).collect::<Vec<_>>();
        let kept = vision::nms_by_class(
            &boxes,
            &scores,
            &classes,
            self.iou_threshold,
            self.max_detections,
        )?;
        Ok(kept
            .into_iter()
            .map(|index| detections[index].clone())
            .collect())
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0. {
        1. / (1. + (-value).exp())
    } else {
        let e = value.exp();
        e / (1. + e)
    }
}

/// Dense results of YOLOX's nondifferentiable SimOTA assignment.
pub struct YoloAssignments {
    /// Matched boxes as `[batch, anchors, 4]` center-x/center-y/width/height.
    pub boxes: Tensor,
    /// Foreground mask as `[batch, anchors, 1]` zeros and ones.
    pub objectness: Tensor,
    /// Matched one-hot classes weighted by IoU as `[batch, anchors, classes]`.
    pub classes: Tensor,
}

/// Weights for the differentiable half of the YOLOX objective.
#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
pub struct YoloLossConfig {
    #[setters(generate)]
    box_weight: f32,
}

impl Default for YoloLossConfig {
    fn default() -> Self {
        Self { box_weight: 5.0 }
    }
}

/// YOLOX loss components before an optimizer transform is attached.
pub struct YoloLoss {
    pub total: Tensor,
    pub boxes: Tensor,
    pub objectness: Tensor,
    pub classes: Tensor,
}

/// Build the YOLOX IoU, objectness and classification objective in RXLA IR.
///
/// SimOTA assignment is deliberately separate because it is nondifferentiable.
/// Forward, these losses, reverse mode, and the optimizer remain one compiled
/// program once its dense assignment tensors are supplied.
pub fn yolox_loss(
    outputs: &[Tensor; 3],
    assignments: &YoloAssignments,
    config: YoloLossConfig,
) -> Result<YoloLoss> {
    let [batch, _, _, channels] =
        <[i64; 4]>::try_from(outputs[0].shape()).map_err(|_| Error::InvalidModel {
            kind: ModelDefinitionError::InvalidYoloAssignments,
        })?;
    let classes = channels - 5;
    let valid_heads = outputs.iter().all(|output| {
        matches!(output.shape(), [b, h, w, c]
            if *b == batch && *h > 0 && *w > 0 && *c == channels)
            && output.dtype() == DType::F32
    });
    let anchors: i64 = outputs
        .iter()
        .map(|output| output.shape()[1] * output.shape()[2])
        .sum();
    if !valid_heads
        || classes <= 0
        || assignments.boxes.shape() != [batch, anchors, 4]
        || assignments.objectness.shape() != [batch, anchors, 1]
        || assignments.classes.shape() != [batch, anchors, classes]
        || assignments.boxes.dtype() != DType::F32
        || assignments.objectness.dtype() != DType::F32
        || assignments.classes.dtype() != DType::F32
    {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidYoloAssignments,
        });
    }

    let mut decoded = Vec::with_capacity(3);
    let mut objectness_logits = Vec::with_capacity(3);
    let mut class_logits = Vec::with_capacity(3);
    for (output, stride) in outputs.iter().zip([8, 16, 32]) {
        let [_, height, width, _] = <[i64; 4]>::try_from(output.shape()).unwrap();
        let x = output.iota_i32(&[1, height, width, 1], 2)?.to_f32()?;
        let y = output.iota_i32(&[1, height, width, 1], 1)?.to_f32()?;
        let grid = Tensor::concatenate(&[x, y], 3)?.broadcast_to([batch, height, width, 2])?;
        let xy = output
            .narrow(3, 0, 2)?
            .add(&grid)?
            .mul_scalar(stride as f32)?;
        let wh = output.narrow(3, 2, 2)?.exp()?.mul_scalar(stride as f32)?;
        decoded.push(Tensor::concatenate(&[xy, wh], 3)?.reshape([batch, height * width, 4])?);
        objectness_logits.push(
            output
                .narrow(3, 4, 1)?
                .reshape([batch, height * width, 1])?,
        );
        class_logits.push(output.narrow(3, 5, classes)?.reshape([
            batch,
            height * width,
            classes,
        ])?);
    }
    let decoded = Tensor::concatenate(&decoded, 1)?;
    let objectness_logits = Tensor::concatenate(&objectness_logits, 1)?;
    let class_logits = Tensor::concatenate(&class_logits, 1)?;
    let foreground = &assignments.objectness;
    let normalizer = foreground
        .sum([0, 1, 2], false)?
        .maximum(&foreground.scalar(1.0)?)?;

    let iou = aligned_cxcywh_iou(&decoded, &assignments.boxes)?;
    let boxes = iou
        .square()?
        .neg()?
        .add_scalar(1.0)?
        .mul(&foreground.reshape([batch, anchors])?)?
        .sum([0, 1], false)?
        .div(&normalizer)?;
    let objectness = objectness_logits
        .binary_cross_entropy_with_logits(foreground)?
        .sum([0, 1, 2], false)?
        .div(&normalizer)?;
    let classes_loss = class_logits
        .binary_cross_entropy_with_logits(&assignments.classes)?
        .mul(&foreground.broadcast_to([batch, anchors, classes])?)?
        .sum([0, 1, 2], false)?
        .div(&normalizer)?;
    let total = boxes
        .mul_scalar(config.box_weight)?
        .add(&objectness)?
        .add(&classes_loss)?;
    Ok(YoloLoss {
        total,
        boxes,
        objectness,
        classes: classes_loss,
    })
}

fn aligned_cxcywh_iou(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let lhs_xy = lhs.narrow(2, 0, 2)?;
    let lhs_half = lhs.narrow(2, 2, 2)?.mul_scalar(0.5)?;
    let rhs_xy = rhs.narrow(2, 0, 2)?;
    let rhs_half = rhs.narrow(2, 2, 2)?.mul_scalar(0.5)?;
    let intersection_min = lhs_xy.sub(&lhs_half)?.maximum(&rhs_xy.sub(&rhs_half)?)?;
    let intersection_max = lhs_xy.add(&lhs_half)?.minimum(&rhs_xy.add(&rhs_half)?)?;
    let intersection_wh = intersection_max
        .sub(&intersection_min)?
        .maximum(&lhs.scalar(0.0)?.broadcast_to(intersection_max.shape())?)?;
    let intersection = pair_product(&intersection_wh)?;
    let lhs_area = pair_product(&lhs.narrow(2, 2, 2)?)?;
    let rhs_area = pair_product(&rhs.narrow(2, 2, 2)?)?;
    Ok(intersection.div(
        &lhs_area
            .add(&rhs_area)?
            .sub(&intersection)?
            .add_scalar(1e-8)?,
    )?)
}

fn pair_product(input: &Tensor) -> Result<Tensor> {
    let shape = [input.shape()[0], input.shape()[1]];
    input
        .narrow(2, 0, 1)?
        .mul(&input.narrow(2, 1, 1)?)?
        .reshape(shape)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::Tracer;
    use rxla_nn::{Model, ModelInput};

    #[test]
    fn tiny_has_official_scales_and_module_paths() {
        let config = YoloConfig::yolox_tiny(80);
        let model = Model::new(|cx: Cx, image| yolo(cx, &image, &config))
            .inputs(ModelInput::new([1, 64, 64, 3]))
            .trace()
            .unwrap();
        assert_eq!(
            model
                .outputs()
                .iter()
                .map(Tensor::shape)
                .collect::<Vec<_>>(),
            [&[1, 8, 8, 85][..], &[1, 4, 4, 85], &[1, 2, 2, 85]]
        );
        for path in [
            "backbone.backbone.stem.conv.conv.weight",
            "backbone.backbone.dark5.1.conv2.conv.weight",
            "backbone.C3_p3.conv3.conv.weight",
            "head.stems.0.conv.weight",
            "head.reg_preds.2.weight",
            "head.obj_preds.2.bias",
            "head.cls_preds.2.weight",
        ] {
            assert!(model.schema().get(path).is_some(), "missing {path}");
        }
        model.prepare_stateful().unwrap();
    }

    #[test]
    fn decode_uses_grid_stride_and_class_aware_nms() {
        let mut small = vec![-20.; 2 * 2 * 7];
        for (cell, class) in [(0, 0), (1, 0), (2, 1)] {
            let row = &mut small[cell * 7..cell * 7 + 7];
            row[..4].fill(0.);
            row[4] = 8.;
            row[5 + class] = 8.;
        }
        let empty = vec![-20.; 7];
        let detections = YoloDecodeConfig::default()
            .confidence(0.5)
            .iou_threshold(0.5)
            .max_detections(10)
            .decode([
                (&small, [1, 2, 2, 7], 8),
                (&empty, [1, 1, 1, 7], 16),
                (&empty, [1, 1, 1, 7], 32),
            ])
            .unwrap();
        assert_eq!(detections.len(), 3);
        assert_eq!(detections[0].xyxy, [-4., -4., 4., 4.]);
        assert_eq!(detections.iter().filter(|v| v.class == 1).count(), 1);
    }

    #[test]
    fn loss_is_scalar_and_differentiable_for_all_heads() {
        let graph = Tracer::default();
        let outputs = [
            graph.input([1, 2, 2, 7]).unwrap(),
            graph.input([1, 1, 1, 7]).unwrap(),
            graph.input([1, 1, 1, 7]).unwrap(),
        ];
        let assignments = YoloAssignments {
            boxes: graph.input([1, 6, 4]).unwrap(),
            objectness: graph.input([1, 6, 1]).unwrap(),
            classes: graph.input([1, 6, 2]).unwrap(),
        };
        let loss = yolox_loss(&outputs, &assignments, YoloLossConfig::default()).unwrap();
        assert!(loss.total.shape().is_empty());
        assert!(loss.boxes.shape().is_empty());
        assert!(loss.objectness.shape().is_empty());
        assert!(loss.classes.shape().is_empty());
        let gradients = loss.total.grad(&outputs).unwrap();
        assert_eq!(gradients.len(), 3);
        for (gradient, output) in gradients.iter().zip(&outputs) {
            assert_eq!(gradient.shape(), output.shape());
        }
    }
}
