//! Explicit host-side vision utilities. No implicit device download or graph
//! execution. These functions are not GPU kernels or differentiable operators.
use crate::{Result, err};

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
) -> Result<Vec<usize>> {
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
) -> Result<Vec<usize>> {
    suppress(boxes, scores, Some(classes), iou_threshold, max_output)
}

fn suppress(
    boxes: &[[f32; 4]],
    scores: &[f32],
    classes: Option<&[i32]>,
    iou_threshold: f32,
    max_output: usize,
) -> Result<Vec<usize>> {
    if classes.is_some_and(|classes| classes.len() != boxes.len()) {
        return Err(err("NMS box/class count mismatch"));
    }
    if boxes.len() != scores.len() {
        return Err(err("NMS box/score count mismatch"));
    }
    if !iou_threshold.is_finite() || !(0. ..=1.).contains(&iou_threshold) {
        return Err(err("NMS IoU threshold must be finite in [0,1]"));
    }
    if scores.iter().any(|score| !score.is_finite())
        || boxes
            .iter()
            .any(|b| b.iter().any(|v| !v.is_finite()) || b[2] < b[0] || b[3] < b[1])
    {
        return Err(err(
            "NMS requires finite scores and ordered finite xyxy boxes",
        ));
    }
    if max_output == 0 {
        return Ok(Vec::new());
    }
    let mut order: Vec<_> = (0..boxes.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        // Validation above makes numeric comparison total, with +0 == -0.
        scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b))
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
