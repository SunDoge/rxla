//! Explicit host-side vision utilities. No implicit device download or graph
//! execution. These functions are not GPU kernels or differentiable operators.
use snafu::{Snafu, ensure};

/// Invalid inputs to host-side nonmaximum suppression.
#[derive(Debug, Snafu, PartialEq)]
#[non_exhaustive]
pub enum NmsError {
    #[snafu(display("NMS received {boxes} boxes but {classes} class IDs"))]
    BoxClassCount { boxes: usize, classes: usize },
    #[snafu(display("NMS received {boxes} boxes but {scores} scores"))]
    BoxScoreCount { boxes: usize, scores: usize },
    #[snafu(display("NMS IoU threshold {threshold} is not finite and in [0, 1]"))]
    InvalidThreshold { threshold: f32 },
    #[snafu(display("NMS score {index} is not finite: {value}"))]
    InvalidScore { index: usize, value: f32 },
    #[snafu(display("NMS box {index} is not finite ordered xyxy: {value:?}"))]
    InvalidBox { index: usize, value: [f32; 4] },
}

pub type NmsResult<T> = std::result::Result<T, NmsError>;

/// Greedy class-agnostic nonmaximum suppression of continuous xyxy boxes.
/// Returns original indices in descending score order, breaking equal scores
/// (including signed zero) by lower original index. Suppresses only when IoU
/// is strictly greater than `iou_threshold`; zero-area boxes have zero IoU.
///
/// Boxes/scores must have equal lengths, finite values and ordered coordinates;
/// degenerate boxes and negative scores are accepted. The threshold must be
/// finite in [0,1]. Validation also runs for zero max_output. For class-aware
/// suppression, use nms_by_class or partition by class explicitly.
///
/// Uses F64 geometry to avoid F32 area overflow for finite F32 coordinates,
/// O(N log N + N*K) time and O(N+K) auxiliary storage, where K is at most
/// max_output. It does not construct an N*N matrix, filter scores, or normalize
/// coordinates. Caller owns all device/host transfers and candidate selection.
pub fn nms(
    boxes: &[[f32; 4]],
    scores: &[f32],
    iou_threshold: f32,
    max_output: usize,
) -> NmsResult<Vec<usize>> {
    suppress(boxes, scores, None, iou_threshold, max_output)
}

/// Class-aware host NMS: only boxes with equal class IDs can suppress each
/// other. Class IDs are opaque i32 labels, including negative values. Returned
/// original indices retain global score order and max_output is a global cap,
/// not a per-class cap. Equal-score ties use the lower original index.
///
/// Uses the same coordinate/score/threshold validation and geometry as nms;
/// class count must also match, even for zero max_output. Coordinates are never
/// shifted to encode class identity. No device download or GPU execution.
pub fn nms_by_class(
    boxes: &[[f32; 4]],
    scores: &[f32],
    classes: &[i32],
    iou_threshold: f32,
    max_output: usize,
) -> NmsResult<Vec<usize>> {
    suppress(boxes, scores, Some(classes), iou_threshold, max_output)
}

fn suppress(
    boxes: &[[f32; 4]],
    scores: &[f32],
    classes: Option<&[i32]>,
    iou_threshold: f32,
    max_output: usize,
) -> NmsResult<Vec<usize>> {
    if let Some(classes) = classes {
        ensure!(
            classes.len() == boxes.len(),
            BoxClassCountSnafu {
                boxes: boxes.len(),
                classes: classes.len(),
            }
        );
    }
    ensure!(
        boxes.len() == scores.len(),
        BoxScoreCountSnafu {
            boxes: boxes.len(),
            scores: scores.len(),
        }
    );
    ensure!(
        iou_threshold.is_finite() && (0. ..=1.).contains(&iou_threshold),
        InvalidThresholdSnafu {
            threshold: iou_threshold,
        }
    );
    for (index, &value) in scores.iter().enumerate() {
        ensure!(value.is_finite(), InvalidScoreSnafu { index, value });
    }
    for (index, &value) in boxes.iter().enumerate() {
        ensure!(
            value.iter().all(|coordinate| coordinate.is_finite())
                && value[2] >= value[0]
                && value[3] >= value[1],
            InvalidBoxSnafu { index, value }
        );
    }
    if max_output == 0 {
        return Ok(Vec::new());
    }
    let mut order: Vec<_> = (0..boxes.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        if scores[a] == scores[b] {
            // Numeric equality deliberately treats both signed zeros as a tie.
            a.cmp(&b)
        } else {
            // total_cmp is panic-free; validation above excludes its NaN ordering.
            scores[b].total_cmp(&scores[a])
        }
    });
    let mut kept = Vec::with_capacity(max_output.min(boxes.len()));
    for index in order {
        if kept.iter().any(|&selected| {
            classes.is_none_or(|classes| classes[index] == classes[selected])
                && overlap(&boxes[index], &boxes[selected]) > f64::from(iou_threshold)
        }) {
            continue;
        }
        kept.push(index);
        if kept.len() == max_output {
            break;
        }
    }
    Ok(kept)
}

fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f64 {
    let a = a.map(f64::from);
    let b = b.map(f64::from);
    let area = |v: &[f64; 4]| (v[2] - v[0]) * (v[3] - v[1]);
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    let union = area(&a) + area(&b) - intersection;
    if union > 0. { intersection / union } else { 0. }
}
