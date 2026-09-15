//! Storage metadata and bounded read-only byte views; no PJRT calls or DLPack.
use super::{InvalidArgumentSnafu, Result};
use smallvec::SmallVec;
use snafu::{OptionExt, ensure};
use std::ops::Range;

type InlineDims = SmallVec<i64, 5>;

/// Concrete nonnegative dimensions. Scalars have rank zero and one element.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(InlineDims);

impl Shape {
    pub fn new(dims: &[i64]) -> Result<Self> {
        ensure!(
            !dims.iter().any(|&d| d < 0),
            InvalidArgumentSnafu {
                message: "negative concrete dimension",
            }
        );
        let shape = Self(InlineDims::from_slice_copy(dims));
        shape.numel()?;
        Ok(shape)
    }
    pub fn as_slice(&self) -> &[i64] {
        &self.0
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn numel(&self) -> Result<usize> {
        if self.0.contains(&0) {
            return Ok(0);
        }
        self.0.iter().try_fold(1usize, |n, &d| {
            usize::try_from(d)
                .ok()
                .and_then(|d| n.checked_mul(d))
                .context(InvalidArgumentSnafu {
                    message: "shape element count overflow",
                })
        })
    }
}

/// Strides in bytes, not elements. Does not imply alignment or non-overlap.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ByteStrides(InlineDims);

fn checked_element_bytes(bytes: usize) -> Result<i64> {
    let bytes = i64::try_from(bytes).ok().context(InvalidArgumentSnafu {
        message: "element size overflow",
    })?;
    ensure!(
        bytes != 0,
        InvalidArgumentSnafu {
            message: "zero element size",
        }
    );
    Ok(bytes)
}

impl ByteStrides {
    pub fn new(strides: &[i64]) -> Self {
        Self(InlineDims::from_slice_copy(strides))
    }
    /// Convert strides measured in elements to strides measured in bytes.
    pub fn from_elements(strides: &[i64], element_bytes: usize) -> Result<Self> {
        let bytes = checked_element_bytes(element_bytes)?;
        let values = strides
            .iter()
            .map(|&s| {
                s.checked_mul(bytes).context(InvalidArgumentSnafu {
                    message: "byte stride overflow",
                })
            })
            .collect::<Result<InlineDims>>()?;
        Ok(Self(values))
    }
    pub fn as_slice(&self) -> &[i64] {
        &self.0
    }
}

/// Concrete conventional byte-strided storage description, not a device view.
/// Tiled/unknown native layouts must not be coerced into this type.
/// Byte offsets are relative to the supplied storage, never raw addresses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StridedLayout {
    shape: Shape,
    strides: ByteStrides,
    byte_offset: u64,
    element_bytes: usize,
    span: Range<usize>,
    payload_bytes: usize,
}

impl StridedLayout {
    pub fn new(
        shape: Shape,
        strides: ByteStrides,
        byte_offset: u64,
        element_bytes: usize,
    ) -> Result<Self> {
        checked_element_bytes(element_bytes)?;
        ensure!(
            shape.rank() == strides.0.len(),
            InvalidArgumentSnafu {
                message: "shape/stride rank mismatch",
            }
        );
        let payload_bytes =
            shape
                .numel()?
                .checked_mul(element_bytes)
                .context(InvalidArgumentSnafu {
                    message: "logical payload size overflow",
                })?;
        let offset = usize::try_from(byte_offset)
            .ok()
            .context(InvalidArgumentSnafu {
                message: "byte offset overflow",
            })?;
        let span = if payload_bytes == 0 {
            offset..offset
        } else {
            let mut low = i128::from(byte_offset);
            let mut high = low;
            for (&d, &s) in shape.0.iter().zip(&strides.0) {
                let delta = i128::from(d - 1) * i128::from(s);
                low = low
                    .checked_add(delta.min(0))
                    .context(InvalidArgumentSnafu {
                        message: "view span overflow",
                    })?;
                high = high
                    .checked_add(delta.max(0))
                    .context(InvalidArgumentSnafu {
                        message: "view span overflow",
                    })?;
            }
            high = high
                .checked_add(element_bytes as i128)
                .context(InvalidArgumentSnafu {
                    message: "view span overflow",
                })?;
            usize::try_from(low).ok().context(InvalidArgumentSnafu {
                message: "view reaches before storage",
            })?..usize::try_from(high).ok().context(InvalidArgumentSnafu {
                message: "view span exceeds address space",
            })?
        };
        Ok(Self {
            shape,
            strides,
            byte_offset,
            element_bytes,
            span,
            payload_bytes,
        })
    }
    /// Canonical row-major layout; empty tensors use all-zero strides.
    pub fn row_major(shape: Shape, element_bytes: usize) -> Result<Self> {
        let mut stride = checked_element_bytes(element_bytes)?;
        let mut strides: InlineDims = std::iter::repeat_n(0, shape.rank()).collect();
        if shape.numel()? != 0 {
            for axis in (0..shape.rank()).rev() {
                strides[axis] = stride;
                if axis != 0 {
                    stride = stride
                        .checked_mul(shape.0[axis])
                        .context(InvalidArgumentSnafu {
                            message: "dense stride overflow",
                        })?;
                }
            }
        }
        Self::new(shape, ByteStrides(strides), 0, element_bytes)
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn byte_strides(&self) -> &ByteStrides {
        &self.strides
    }
    pub fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
    pub fn element_bytes(&self) -> usize {
        self.element_bytes
    }
    pub fn logical_payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    /// Bounding interval of addressed bytes, including gaps; empty means no reads.
    pub fn byte_span(&self) -> Range<usize> {
        self.span.clone()
    }
    /// Ignores singleton-axis strides. Empty tensors are contiguous by convention.
    pub fn is_row_contiguous(&self) -> bool {
        if self.payload_bytes == 0 {
            return true;
        }
        let mut expected = self.element_bytes as i128;
        for (&d, &s) in self.shape.0.iter().zip(&self.strides.0).rev() {
            if d > 1 && i128::from(s) != expected {
                return false;
            }
            expected *= i128::from(d);
        }
        true
    }
    fn element_offset(&self, indices: &[i64]) -> Result<usize> {
        ensure!(
            indices.len() == self.shape.rank(),
            InvalidArgumentSnafu {
                message: "index rank mismatch",
            }
        );
        let mut offset = i128::from(self.byte_offset);
        for ((&i, &d), &s) in indices.iter().zip(&self.shape.0).zip(&self.strides.0) {
            ensure!(
                i >= 0 && i < d,
                InvalidArgumentSnafu {
                    message: "view index out of bounds",
                }
            );
            offset += i128::from(i) * i128::from(s);
        }
        // Constructor proves the complete reachable interval fits usize.
        Ok(offset as usize)
    }
}

/// Borrowed read-only raw bytes. Overlapping/broadcast views are allowed;
/// no typed references, alignment promise, mutable alias or native handle exists.
pub struct HostView<'a> {
    storage: &'a [u8],
    layout: StridedLayout,
}
impl<'a> HostView<'a> {
    pub fn new(storage: &'a [u8], layout: StridedLayout) -> Result<Self> {
        ensure!(
            layout.span.end <= storage.len(),
            InvalidArgumentSnafu {
                message: "view exceeds supplied storage",
            }
        );
        Ok(Self { storage, layout })
    }
    pub fn layout(&self) -> &StridedLayout {
        &self.layout
    }
    /// Length of the supplied storage slice, not the original allocation capacity.
    pub fn storage_bytes(&self) -> usize {
        self.storage.len()
    }
    pub fn element(&self, indices: &[i64]) -> Result<&'a [u8]> {
        let offset = self.layout.element_offset(indices)?;
        Ok(&self.storage[offset..offset + self.layout.element_bytes])
    }
    /// Explicit row-major packing into caller-owned storage. No allocation.
    pub fn copy_to_contiguous(&self, output: &mut [u8]) -> Result<()> {
        ensure!(
            output.len() == self.layout.payload_bytes,
            InvalidArgumentSnafu {
                message: "packed destination size mismatch",
            }
        );
        let mut indices: InlineDims = std::iter::repeat_n(0, self.layout.shape.rank()).collect();
        for chunk in output.chunks_exact_mut(self.layout.element_bytes) {
            chunk.copy_from_slice(self.element(&indices)?);
            for axis in (0..indices.len()).rev() {
                indices[axis] += 1;
                if indices[axis] < self.layout.shape.0[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inline_storage_and_units() {
        let five = Shape::new(&[1; 5]).unwrap();
        assert!(!five.0.spilled());
        assert!(Shape::new(&[1; 6]).unwrap().0.spilled());
        let bytes = ByteStrides::from_elements(&[-3, 0, 1], 4).unwrap();
        assert!(!bytes.0.spilled());
        assert_eq!(bytes.as_slice(), [-12, 0, 4]);
        assert!(ByteStrides::from_elements(&[1], 0).is_err());
        assert!(ByteStrides::from_elements(&[i64::MAX], 2).is_err());
    }
    #[test]
    fn transpose_reverse_broadcast_and_step_pack() {
        for (shape, strides, offset, expected) in [
            (vec![3, 2], vec![1, 3], 0, vec![0, 3, 1, 4, 2, 5]),
            (vec![6], vec![-1], 5, vec![5, 4, 3, 2, 1, 0]),
            (vec![2, 3], vec![0, 1], 0, vec![0, 1, 2, 0, 1, 2]),
            (vec![3], vec![2], 0, vec![0, 2, 4]),
        ] {
            let layout = StridedLayout::new(
                Shape::new(&shape).unwrap(),
                ByteStrides::new(&strides),
                offset,
                1,
            )
            .unwrap();
            assert!(!layout.is_row_contiguous());
            let view = HostView::new(&[0, 1, 2, 3, 4, 5], layout).unwrap();
            let mut packed = vec![0; expected.len()];
            view.copy_to_contiguous(&mut packed).unwrap();
            assert_eq!(packed, expected);
        }
    }
    #[test]
    fn scalar_empty_singletons_and_multi_byte_elements() {
        let scalar = StridedLayout::row_major(Shape::new(&[]).unwrap(), 2).unwrap();
        assert_eq!(scalar.byte_span(), 0..2);
        assert_eq!(
            HostView::new(&[1, 2], scalar)
                .unwrap()
                .element(&[])
                .unwrap(),
            [1, 2]
        );
        let empty =
            StridedLayout::row_major(Shape::new(&[i64::MAX, 0, i64::MAX]).unwrap(), 4).unwrap();
        assert_eq!(empty.logical_payload_bytes(), 0);
        HostView::new(&[], empty)
            .unwrap()
            .copy_to_contiguous(&mut [])
            .unwrap();
        let layout = StridedLayout::new(
            Shape::new(&[1, 2, 1]).unwrap(),
            ByteStrides::new(&[-99, 2, 0]),
            2,
            2,
        )
        .unwrap();
        assert!(layout.is_row_contiguous());
        assert_eq!(layout.byte_span(), 2..6);
        let view = HostView::new(&[9, 9, 1, 2, 3, 4], layout).unwrap();
        let mut output = [0; 4];
        view.copy_to_contiguous(&mut output).unwrap();
        assert_eq!(output, [1, 2, 3, 4]);
    }
    #[test]
    fn exhaustive_small_views_match_enumerated_addresses() {
        let storage: Vec<u8> = (0..16).collect();
        for rows in 0..4 {
            for cols in 0..4 {
                for row_stride in -3..4 {
                    for col_stride in -3..4 {
                        for offset in 0..8 {
                            let mut addresses = Vec::new();
                            for row in 0..rows {
                                for col in 0..cols {
                                    addresses.push(offset + row * row_stride + col * col_stride);
                                }
                            }
                            let valid = addresses.iter().all(|&a| (0..16).contains(&a));
                            let view = StridedLayout::new(
                                Shape::new(&[rows, cols]).unwrap(),
                                ByteStrides::new(&[row_stride, col_stride]),
                                offset as u64,
                                1,
                            )
                            .and_then(|layout| HostView::new(&storage, layout));
                            assert_eq!(view.is_ok(), valid);
                            if let Ok(view) = view {
                                let expected_span =
                                    match (addresses.iter().min(), addresses.iter().max()) {
                                        (Some(&lo), Some(&hi)) => lo as usize..hi as usize + 1,
                                        _ => offset as usize..offset as usize,
                                    };
                                assert_eq!(view.layout().byte_span(), expected_span);
                                let mut packed = vec![0; addresses.len()];
                                view.copy_to_contiguous(&mut packed).unwrap();
                                assert_eq!(
                                    packed,
                                    addresses
                                        .iter()
                                        .map(|&a| storage[a as usize])
                                        .collect::<Vec<_>>()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn validation_rejects_bad_metadata_and_bounds() {
        assert!(Shape::new(&[-1]).is_err());
        assert!(Shape::new(&[i64::MAX, i64::MAX]).is_err());
        let shape = Shape::new(&[2]).unwrap();
        assert!(StridedLayout::new(shape.clone(), ByteStrides::new(&[]), 0, 1).is_err());
        assert!(StridedLayout::new(shape.clone(), ByteStrides::new(&[-1]), 0, 1).is_err());
        assert!(StridedLayout::new(shape.clone(), ByteStrides::new(&[1]), u64::MAX, 1).is_err());
        let layout = StridedLayout::row_major(shape, 4).unwrap();
        assert!(HostView::new(&[0; 7], layout.clone()).is_err());
        let view = HostView::new(&[0; 8], layout).unwrap();
        assert!(view.element(&[-1]).is_err());
        assert!(view.element(&[2]).is_err());
        assert!(view.element(&[]).is_err());
        assert!(view.copy_to_contiguous(&mut [0; 7]).is_err());
    }
}
