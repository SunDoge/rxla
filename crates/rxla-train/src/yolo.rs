//! Host-side, no-gradient YOLOX SimOTA assignment.

use derive_setters::Setters;
use snafu::{Snafu, ensure};

pub type SimOtaResult<T, E = SimOtaError> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum SimOtaError {
    #[snafu(display("invalid SimOTA prediction buffers or dimensions"))]
    Predictions,
    #[snafu(display("invalid SimOTA target at index {index}"))]
    Target { index: usize },
    #[snafu(display("invalid SimOTA configuration"))]
    Configuration,
}

/// One raw NHWC YOLOX output head.
#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
pub struct YoloHead<'a> {
    values: &'a [f32],
    shape: [usize; 4],
    #[setters(generate)]
    stride: usize,
}

impl<'a> YoloHead<'a> {
    pub fn new(values: &'a [f32], shape: [usize; 4]) -> Self {
        Self {
            values,
            shape,
            stride: 0,
        }
    }
}

/// Owned decoded predictions and anchor metadata ready for SimOTA.
pub struct DecodedYoloBatch {
    boxes: Vec<[f32; 4]>,
    objectness_logits: Vec<f32>,
    class_logits: Vec<f32>,
    anchor_centers: Vec<[f32; 2]>,
    anchor_strides: Vec<f32>,
    batch: usize,
    anchors: usize,
    classes: usize,
}

impl DecodedYoloBatch {
    pub fn from_heads(heads: [YoloHead<'_>; 3]) -> SimOtaResult<Self> {
        let batch = heads[0].shape[0];
        let channels = heads[0].shape[3];
        ensure!(batch > 0 && channels >= 6, PredictionsSnafu);
        let classes = channels - 5;
        let anchors = heads.iter().map(|head| head.shape[1] * head.shape[2]).sum();
        ensure!(
            heads.iter().all(|head| {
                head.shape[0] == batch
                    && head.shape[3] == channels
                    && head.stride > 0
                    && head.values.len() == head.shape.iter().product()
            }),
            PredictionsSnafu
        );
        let mut boxes = vec![[0.0; 4]; batch * anchors];
        let mut objectness_logits = vec![0.0; batch * anchors];
        let mut class_logits = vec![0.0; batch * anchors * classes];
        let mut anchor_centers = Vec::with_capacity(anchors);
        let mut anchor_strides = Vec::with_capacity(anchors);
        let mut anchor_offset = 0;
        for head in heads {
            let [_, height, width, _] = head.shape;
            for y in 0..height {
                for x in 0..width {
                    anchor_centers.push([
                        (x as f32 + 0.5) * head.stride as f32,
                        (y as f32 + 0.5) * head.stride as f32,
                    ]);
                    anchor_strides.push(head.stride as f32);
                }
            }
            let head_anchors = height * width;
            for image in 0..batch {
                for anchor in 0..head_anchors {
                    let source = (image * head_anchors + anchor) * channels;
                    let x = anchor % width;
                    let y = anchor / width;
                    let destination = image * anchors + anchor_offset + anchor;
                    boxes[destination] = [
                        (head.values[source] + x as f32) * head.stride as f32,
                        (head.values[source + 1] + y as f32) * head.stride as f32,
                        head.values[source + 2].exp() * head.stride as f32,
                        head.values[source + 3].exp() * head.stride as f32,
                    ];
                    objectness_logits[destination] = head.values[source + 4];
                    class_logits[destination * classes..(destination + 1) * classes]
                        .copy_from_slice(&head.values[source + 5..source + channels]);
                }
            }
            anchor_offset += head_anchors;
        }
        Ok(Self {
            boxes,
            objectness_logits,
            class_logits,
            anchor_centers,
            anchor_strides,
            batch,
            anchors,
            classes,
        })
    }

    pub fn input(&self) -> SimOtaInput<'_> {
        SimOtaInput {
            boxes: &self.boxes,
            objectness_logits: &self.objectness_logits,
            class_logits: &self.class_logits,
            anchor_centers: &self.anchor_centers,
            anchor_strides: &self.anchor_strides,
            batch: self.batch,
            anchors: self.anchors,
            classes: self.classes,
        }
    }
}

/// One ground-truth detection in input-pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionTarget {
    pub batch: usize,
    pub class: usize,
    pub cxcywh: [f32; 4],
}

/// Contiguous decoded predictions passed to the no-gradient matcher.
pub struct SimOtaInput<'a> {
    pub boxes: &'a [[f32; 4]],
    pub objectness_logits: &'a [f32],
    pub class_logits: &'a [f32],
    pub anchor_centers: &'a [[f32; 2]],
    pub anchor_strides: &'a [f32],
    pub batch: usize,
    pub anchors: usize,
    pub classes: usize,
}

/// Dense targets consumed by `rxla_models::yolox_loss`.
#[derive(Clone, Debug, PartialEq)]
pub struct SimOtaAssignments {
    pub boxes: Vec<f32>,
    pub objectness: Vec<f32>,
    pub classes: Vec<f32>,
    pub batch: usize,
    pub anchors: usize,
    pub class_count: usize,
    pub foreground: usize,
}

/// Official YOLOX matching policy with explicit tunable constants.
#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
pub struct SimOtaConfig {
    #[setters(generate)]
    center_radius: f32,
    #[setters(generate)]
    candidate_top_k: usize,
    #[setters(generate)]
    iou_weight: f32,
    #[setters(generate)]
    outside_penalty: f32,
}

/// YOLOX's quadratic-warmup, cosine-decay, minimum-rate-tail schedule.
#[derive(Clone, Copy, Debug, Setters)]
#[setters(generate = false)]
pub struct YoloLrSchedule {
    base_learning_rate: f32,
    total_steps: usize,
    #[setters(generate)]
    warmup_steps: usize,
    #[setters(generate)]
    no_augmentation_steps: usize,
    #[setters(generate)]
    warmup_learning_rate: f32,
    #[setters(generate)]
    minimum_ratio: f32,
}

impl YoloLrSchedule {
    pub fn new(base_learning_rate: f32, total_steps: usize) -> Self {
        Self {
            base_learning_rate,
            total_steps,
            warmup_steps: 0,
            no_augmentation_steps: 0,
            warmup_learning_rate: 0.0,
            minimum_ratio: 0.05,
        }
    }

    pub fn at(self, step: usize) -> SimOtaResult<f32> {
        ensure!(
            self.base_learning_rate.is_finite()
                && self.base_learning_rate >= 0.0
                && self.total_steps > 0
                && self.warmup_steps + self.no_augmentation_steps <= self.total_steps
                && self.warmup_learning_rate.is_finite()
                && self.warmup_learning_rate >= 0.0
                && self.minimum_ratio.is_finite()
                && (0.0..=1.0).contains(&self.minimum_ratio)
                && step < self.total_steps,
            ConfigurationSnafu
        );
        let minimum = self.base_learning_rate * self.minimum_ratio;
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let progress = step as f32 / self.warmup_steps as f32;
            return Ok(
                (self.base_learning_rate - self.warmup_learning_rate) * progress * progress
                    + self.warmup_learning_rate,
            );
        }
        let cosine_steps = self
            .total_steps
            .saturating_sub(self.warmup_steps + self.no_augmentation_steps);
        if cosine_steps == 0 || step >= self.total_steps - self.no_augmentation_steps {
            return Ok(minimum);
        }
        let progress = (step - self.warmup_steps) as f32 / cosine_steps as f32;
        Ok(minimum
            + 0.5
                * (self.base_learning_rate - minimum)
                * (1.0 + (std::f32::consts::PI * progress).cos()))
    }
}

impl Default for SimOtaConfig {
    fn default() -> Self {
        Self {
            center_radius: 2.5,
            candidate_top_k: 10,
            iou_weight: 3.0,
            outside_penalty: 100_000.0,
        }
    }
}

impl SimOtaConfig {
    /// Match predictions to targets using YOLOX dynamic-k assignment.
    pub fn assign(
        self,
        input: &SimOtaInput<'_>,
        targets: &[DetectionTarget],
    ) -> SimOtaResult<SimOtaAssignments> {
        self.validate(input, targets)?;
        let mut output = SimOtaAssignments {
            boxes: vec![0.0; input.batch * input.anchors * 4],
            objectness: vec![0.0; input.batch * input.anchors],
            classes: vec![0.0; input.batch * input.anchors * input.classes],
            batch: input.batch,
            anchors: input.anchors,
            class_count: input.classes,
            foreground: 0,
        };
        for batch in 0..input.batch {
            let ground_truth = targets
                .iter()
                .filter(|target| target.batch == batch)
                .collect::<Vec<_>>();
            self.assign_image(input, batch, &ground_truth, &mut output);
        }
        Ok(output)
    }

    fn validate(self, input: &SimOtaInput<'_>, targets: &[DetectionTarget]) -> SimOtaResult<()> {
        ensure!(
            self.center_radius.is_finite()
                && self.center_radius > 0.0
                && self.candidate_top_k > 0
                && self.iou_weight.is_finite()
                && self.iou_weight >= 0.0
                && self.outside_penalty.is_finite()
                && self.outside_penalty >= 0.0,
            ConfigurationSnafu
        );
        ensure!(
            input.batch > 0
                && input.anchors > 0
                && input.classes > 0
                && input.boxes.len() == input.batch * input.anchors
                && input.objectness_logits.len() == input.batch * input.anchors
                && input.class_logits.len() == input.batch * input.anchors * input.classes
                && input.anchor_centers.len() == input.anchors
                && input.anchor_strides.len() == input.anchors
                && input.boxes.iter().flatten().all(|value| value.is_finite())
                && input
                    .objectness_logits
                    .iter()
                    .all(|value| value.is_finite())
                && input.class_logits.iter().all(|value| value.is_finite())
                && input
                    .anchor_centers
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                && input
                    .anchor_strides
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
            PredictionsSnafu
        );
        for (index, target) in targets.iter().enumerate() {
            ensure!(
                target.batch < input.batch
                    && target.class < input.classes
                    && target.cxcywh.iter().all(|value| value.is_finite())
                    && target.cxcywh[2] > 0.0
                    && target.cxcywh[3] > 0.0,
                TargetSnafu { index }
            );
        }
        Ok(())
    }

    fn assign_image(
        self,
        input: &SimOtaInput<'_>,
        batch: usize,
        targets: &[&DetectionTarget],
        output: &mut SimOtaAssignments,
    ) {
        if targets.is_empty() {
            return;
        }
        let candidates = (0..input.anchors)
            .filter_map(|anchor| {
                let mut inside_either = false;
                let mut inside_both = Vec::with_capacity(targets.len());
                for target in targets {
                    let in_box = contains(target.cxcywh, input.anchor_centers[anchor]);
                    let radius = self.center_radius * input.anchor_strides[anchor];
                    let center_box = [
                        target.cxcywh[0],
                        target.cxcywh[1],
                        radius * 2.0,
                        radius * 2.0,
                    ];
                    let in_center = contains(center_box, input.anchor_centers[anchor]);
                    inside_either |= in_box || in_center;
                    inside_both.push(in_box && in_center);
                }
                inside_either.then_some((anchor, inside_both))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return;
        }

        let mut costs = vec![vec![0.0; candidates.len()]; targets.len()];
        let mut ious = vec![vec![0.0; candidates.len()]; targets.len()];
        for (target_index, target) in targets.iter().enumerate() {
            for (candidate_index, (anchor, inside_both)) in candidates.iter().enumerate() {
                let prediction = input.boxes[batch * input.anchors + anchor];
                let iou = box_iou(target.cxcywh, prediction);
                ious[target_index][candidate_index] = iou;
                let base = (batch * input.anchors + anchor) * input.classes;
                let objectness = sigmoid(input.objectness_logits[batch * input.anchors + anchor]);
                let classification = (0..input.classes)
                    .map(|class| {
                        let probability =
                            (sigmoid(input.class_logits[base + class]) * objectness).sqrt();
                        binary_cross_entropy(probability, usize::from(class == target.class) as f32)
                    })
                    .sum::<f32>();
                costs[target_index][candidate_index] = classification
                    - self.iou_weight * (iou + 1e-8).ln()
                    + if inside_both[target_index] {
                        0.0
                    } else {
                        self.outside_penalty
                    };
            }
        }

        let mut matching = vec![vec![false; candidates.len()]; targets.len()];
        for target in 0..targets.len() {
            let mut ranked_ious = ious[target].clone();
            ranked_ious.sort_by(|left, right| right.total_cmp(left));
            let dynamic_k = ranked_ious
                .iter()
                .take(self.candidate_top_k)
                .sum::<f32>()
                .floor()
                .max(1.0) as usize;
            let mut ranked_costs = (0..candidates.len()).collect::<Vec<_>>();
            ranked_costs
                .sort_by(|left, right| costs[target][*left].total_cmp(&costs[target][*right]));
            for &candidate in ranked_costs.iter().take(dynamic_k.min(candidates.len())) {
                matching[target][candidate] = true;
            }
        }

        for (candidate, (anchor, _)) in candidates.iter().enumerate() {
            let matched = (0..targets.len())
                .filter(|target| matching[*target][candidate])
                .min_by(|left, right| costs[*left][candidate].total_cmp(&costs[*right][candidate]));
            let Some(target_index) = matched else {
                continue;
            };
            let target = targets[target_index];
            let iou = ious[target_index][candidate];
            let flat_anchor = batch * input.anchors + anchor;
            output.boxes[flat_anchor * 4..flat_anchor * 4 + 4].copy_from_slice(&target.cxcywh);
            output.objectness[flat_anchor] = 1.0;
            output.classes[flat_anchor * input.classes + target.class] = iou;
            output.foreground += 1;
        }
    }
}

fn contains(cxcywh: [f32; 4], point: [f32; 2]) -> bool {
    let half_width = cxcywh[2] * 0.5;
    let half_height = cxcywh[3] * 0.5;
    point[0] > cxcywh[0] - half_width
        && point[0] < cxcywh[0] + half_width
        && point[1] > cxcywh[1] - half_height
        && point[1] < cxcywh[1] + half_height
}

fn box_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection_width = ((left[0] + left[2] * 0.5).min(right[0] + right[2] * 0.5)
        - (left[0] - left[2] * 0.5).max(right[0] - right[2] * 0.5))
    .max(0.0);
    let intersection_height = ((left[1] + left[3] * 0.5).min(right[1] + right[3] * 0.5)
        - (left[1] - left[3] * 0.5).max(right[1] - right[3] * 0.5))
    .max(0.0);
    let intersection = intersection_width * intersection_height;
    intersection / (left[2] * left[3] + right[2] * right[3] - intersection + 1e-8)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn binary_cross_entropy(probability: f32, target: f32) -> f32 {
    let probability = probability.clamp(1e-8, 1.0 - 1e-8);
    -target * probability.ln() - (1.0 - target) * (-probability).ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_matching_selects_best_anchor_and_resolves_conflicts() {
        let boxes = [
            [10.0, 10.0, 8.0, 8.0],
            [11.0, 10.0, 8.0, 8.0],
            [40.0, 40.0, 8.0, 8.0],
        ];
        let input = SimOtaInput {
            boxes: &boxes,
            objectness_logits: &[8.0; 3],
            class_logits: &[8.0, -8.0, 7.0, -7.0, -8.0, 8.0],
            anchor_centers: &[[10.0, 10.0], [12.0, 10.0], [40.0, 40.0]],
            anchor_strides: &[8.0; 3],
            batch: 1,
            anchors: 3,
            classes: 2,
        };
        let targets = [
            DetectionTarget {
                batch: 0,
                class: 0,
                cxcywh: [10.0, 10.0, 8.0, 8.0],
            },
            DetectionTarget {
                batch: 0,
                class: 1,
                cxcywh: [40.0, 40.0, 8.0, 8.0],
            },
        ];
        let assigned = SimOtaConfig::default().assign(&input, &targets).unwrap();
        assert_eq!(assigned.foreground, 2);
        assert_eq!(assigned.objectness, [1.0, 0.0, 1.0]);
        assert_eq!(assigned.classes[0], 1.0);
        assert_eq!(assigned.classes[5], 1.0);
    }

    #[test]
    fn empty_image_produces_background_objectness() {
        let input = SimOtaInput {
            boxes: &[[0.0; 4]],
            objectness_logits: &[0.0],
            class_logits: &[0.0],
            anchor_centers: &[[4.0, 4.0]],
            anchor_strides: &[8.0],
            batch: 1,
            anchors: 1,
            classes: 1,
        };
        let assigned = SimOtaConfig::default().assign(&input, &[]).unwrap();
        assert_eq!(assigned.foreground, 0);
        assert_eq!(assigned.objectness, [0.0]);
    }

    #[test]
    fn decodes_three_heads_in_batch_major_anchor_order() {
        let first = [0.0, 0.0, 0.0, 0.0, 3.0, 4.0];
        let second = [1.0, 2.0, 0.0, 0.0, 5.0, 6.0];
        let third = [-1.0, -2.0, 0.0, 0.0, 7.0, 8.0];
        let decoded = DecodedYoloBatch::from_heads([
            YoloHead::new(&first, [1, 1, 1, 6]).stride(8),
            YoloHead::new(&second, [1, 1, 1, 6]).stride(16),
            YoloHead::new(&third, [1, 1, 1, 6]).stride(32),
        ])
        .unwrap();
        let input = decoded.input();
        assert_eq!(input.anchors, 3);
        assert_eq!(input.boxes[0], [0.0, 0.0, 8.0, 8.0]);
        assert_eq!(input.boxes[1], [16.0, 32.0, 16.0, 16.0]);
        assert_eq!(input.boxes[2], [-32.0, -64.0, 32.0, 32.0]);
        assert_eq!(input.anchor_centers, [[4.0, 4.0], [8.0, 8.0], [16.0, 16.0]]);
        assert_eq!(input.class_logits, [4.0, 6.0, 8.0]);
    }

    #[test]
    fn yolox_schedule_has_warmup_cosine_and_minimum_tail() {
        let schedule = YoloLrSchedule::new(1.0, 100)
            .warmup_steps(10)
            .no_augmentation_steps(10)
            .minimum_ratio(0.05);
        assert_eq!(schedule.at(0).unwrap(), 0.0);
        assert!((schedule.at(5).unwrap() - 0.25).abs() < 1e-6);
        assert!((schedule.at(10).unwrap() - 1.0).abs() < 1e-6);
        assert!((schedule.at(50).unwrap() - 0.525).abs() < 1e-6);
        assert_eq!(schedule.at(90).unwrap(), 0.05);
        assert_eq!(schedule.at(99).unwrap(), 0.05);
        assert!(schedule.at(100).is_err());
    }
}
