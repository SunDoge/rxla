use crate::{Result, Tensor, err};

fn coordinates(boxes: &Tensor) -> Result<Vec<Tensor>> {
    if boxes.shape().last() != Some(&4) {
        return Err(err("boxes require a final dimension of four coordinates"));
    }
    boxes.unbind(boxes.ndim() - 1)
}

impl Tensor {
    /// Convert [..., 4] (center_x, center_y, width, height) to xyxy corners.
    /// Preserves shape and gradients. Signed widths/heights are not clamped:
    /// constrain them explicitly (for example with softplus) when ordered boxes
    /// are required. No pixel-inclusive convention or image-size normalization.
    pub fn box_cxcywh_to_xyxy(&self) -> Result<Self> {
        let c = coordinates(self)?;
        let half_width = c[2].mul_scalar(0.5)?;
        let half_height = c[3].mul_scalar(0.5)?;
        Self::stack(
            &[
                c[0].sub(&half_width)?,
                c[1].sub(&half_height)?,
                c[0].add(&half_width)?,
                c[1].add(&half_height)?,
            ],
            self.ndim() - 1,
        )
    }

    /// Convert [..., 4] xyxy corners to (center_x, center_y, width, height).
    /// Preserves signed widths/heights and gradients, without coordinate
    /// reordering, clamping, or host transfer. Finite arithmetic is required;
    /// floating-point round trips need not recover the original bits.
    pub fn box_xyxy_to_cxcywh(&self) -> Result<Self> {
        let c = coordinates(self)?;
        Self::stack(
            &[
                c[0].mul_scalar(0.5)?.add(&c[2].mul_scalar(0.5)?)?,
                c[1].mul_scalar(0.5)?.add(&c[3].mul_scalar(0.5)?)?,
                c[2].sub(&c[0])?,
                c[3].sub(&c[1])?,
            ],
            self.ndim() - 1,
        )
    }

    /// Continuous area for xyxy boxes shaped [..., 4], yielding [...].
    /// Width/height are clamped at zero, so inverted or degenerate boxes have
    /// zero area. No inclusive-pixel +1 convention or coordinate normalization.
    /// Coordinates and intermediate arithmetic should be finite. Composes graph
    /// operators; gradients use the existing ReLU subgradient at zero.
    pub fn box_area(&self) -> Result<Self> {
        let c = coordinates(self)?;
        c[2].sub(&c[0])?.relu()?.mul(&c[3].sub(&c[1])?.relu()?)
    }

    /// Pairwise continuous xyxy IoU: [..., N, 4] and [..., M, 4] produce
    /// broadcast_batch + [N, M]. Both operands must have rank at least two.
    /// Empty box sets are supported. Inverted dimensions clamp to zero; a
    /// zero union returns zero using a safe denominator, without adding epsilon
    /// to valid unions. Finite coordinates/areas are required; overflow is not
    /// repaired. Gradients compose existing min/max/ReLU rules and are piecewise
    /// differentiable, not smooth at coincident edges. No NMS or host transfer.
    pub fn pairwise_box_iou(&self, other: &Self) -> Result<Self> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(err("pairwise_box_iou requires rank >= 2"));
        }
        if self.shape().last() != Some(&4) || other.shape().last() != Some(&4) {
            return Err(err(
                "boxes require a final dimension of four xyxy coordinates",
            ));
        }
        let a = self.unsqueeze(self.ndim() - 1)?;
        let b = other.unsqueeze(other.ndim() - 2)?;
        a.box_iou(&b)
    }

    /// Elementwise continuous xyxy IoU for broadcast-compatible [..., 4]
    /// operands. Returns their broadcast prefix, without inserting pairwise
    /// axes. For [N, 4] predictions and targets this produces [N], not [N, N].
    /// A single [4] box can broadcast across a batch. Uses the same finite-value,
    /// degenerate-box and piecewise-gradient rules as pairwise_box_iou.
    pub fn box_iou(&self, other: &Self) -> Result<Self> {
        self.box_overlap(other, false)
    }

    /// Elementwise generalized IoU for broadcast-compatible [..., 4] xyxy
    /// boxes: IoU - (enclosing_area - union) / enclosing_area. Unlike ordinary
    /// IoU, the enclosure term can supply gradients for disjoint boxes.
    /// Caller must supply ordered finite coordinates (x2 >= x1, y2 >= y1) and
    /// finite intermediate areas; no coordinate reordering is performed.
    /// Zero union/enclosure use safe denominators, with zero contribution for
    /// zero area. Empty sets work. Min/max boundary gradients are piecewise.
    /// See https://giou.stanford.edu/ for the nondegenerate box definition.
    pub fn box_giou(&self, other: &Self) -> Result<Self> {
        self.box_overlap(other, true)
    }

    fn box_overlap(&self, other: &Self, generalized: bool) -> Result<Self> {
        if self.shape().last() != Some(&4) || other.shape().last() != Some(&4) {
            return Err(err(
                "boxes require a final dimension of four xyxy coordinates",
            ));
        }
        let a = self;
        let b = other;
        let rank = a.ndim().max(b.ndim());
        let mut shape = vec![1; rank];
        for (offset, dimension) in shape.iter_mut().rev().enumerate() {
            let x = a.shape().iter().rev().nth(offset).copied().unwrap_or(1);
            let y = b.shape().iter().rev().nth(offset).copied().unwrap_or(1);
            *dimension = if x == y || y == 1 {
                x
            } else if x == 1 {
                y
            } else {
                return Err(err("incompatible box batch dimensions"));
            };
        }
        let a = a.broadcast_to(&shape)?;
        let b = b.broadcast_to(&shape)?;
        let area_a = a.box_area()?;
        let area_b = b.box_area()?;
        let a = coordinates(&a)?;
        let b = coordinates(&b)?;
        let width = a[2].minimum(&b[2])?.sub(&a[0].maximum(&b[0])?)?.relu()?;
        let height = a[3].minimum(&b[3])?.sub(&a[1].maximum(&b[1])?)?.relu()?;
        let intersection = width.mul(&height)?;
        let union = area_a.add(&area_b)?.sub(&intersection)?;
        let positive = union.gt_mask(
            &self
                .graph()
                .constant(&[], &[0.])?
                .broadcast_to(union.shape())?,
        )?;
        let one = self
            .graph()
            .constant(&[], &[1.])?
            .broadcast_to(union.shape())?;
        let iou = intersection.div(&positive.select(&union, &one)?)?;
        if !generalized {
            return Ok(iou);
        }
        let width = a[2].maximum(&b[2])?.sub(&a[0].minimum(&b[0])?)?.relu()?;
        let height = a[3].maximum(&b[3])?.sub(&a[1].minimum(&b[1])?)?.relu()?;
        let enclosure = width.mul(&height)?;
        let nonempty = enclosure.gt_mask(
            &self
                .graph()
                .constant(&[], &[0.])?
                .broadcast_to(enclosure.shape())?,
        )?;
        let penalty = enclosure
            .sub(&union)?
            .div(&nonempty.select(&enclosure, &one)?)?;
        iou.sub(&penalty)
    }
}
