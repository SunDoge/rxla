//! Ownership-preserving DLPack interchange.
//!
//! Import retains the producer's managed tensor and therefore invokes its
//! deleter exactly once when the last rxla storage owner is dropped.

use crate::{DType, Storage, Tensor};
use dlpark::{Managed, ManagedTensorBase, ffi, legacy, versioned};
use snafu::{ResultExt, Snafu, ensure};
use std::ptr::NonNull;

/// DLPack import failures.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ImportError {
    #[snafu(display("invalid DLPack tensor: {source}"))]
    InvalidDescriptor { source: dlpark::tensor::Error },
    #[snafu(display("DLPack device {device_type:?} is not CPU host memory"))]
    UnsupportedDevice { device_type: ffi::DLDeviceType },
    #[snafu(display("unsupported DLPack dtype {dtype:?}"))]
    UnsupportedDType { dtype: ffi::DLDataType },
    #[snafu(display("DLPack tensors with lanes other than one are unsupported"))]
    VectorDType,
    #[snafu(display("non-compact DLPack strides are not yet importable"))]
    NonCompact,
    #[snafu(display("invalid DLPack data pointer: {source}"))]
    InvalidData { source: dlpark::tensor::Error },
    #[snafu(display("DLPack tensor cannot be represented by rxla host storage: {source}"))]
    InvalidTensor { source: crate::TensorBuildError },
}

type Result<T> = std::result::Result<T, ImportError>;

struct Owner<M: ManagedTensorBase> {
    _managed: Managed<M>,
    data: NonNull<u8>,
    len: usize,
}

// SAFETY: import consumes the managed tensor, validates CPU storage, and RXLA
// exposes the allocation only through immutable byte views. The DLPack deleter
// owns the transferred allocation and may run after the handle moves threads.
unsafe impl<M: ManagedTensorBase> Send for Owner<M> {}
unsafe impl<M: ManagedTensorBase> Sync for Owner<M> {}

impl<M: ManagedTensorBase> AsRef<[u8]> for Owner<M> {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the Managed handle retains the producer allocation and its
        // deleter; import validated this compact CPU span before constructing us.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

fn dtype_from_dlpack(dtype: ffi::DLDataType) -> Result<DType> {
    ensure!(dtype.lanes == 1, VectorDTypeSnafu);
    match (dtype.code, dtype.bits) {
        (ffi::DLDataTypeCode::UINT, 8) => Ok(DType::U8),
        (ffi::DLDataTypeCode::FLOAT, 16) => Ok(DType::F16),
        (ffi::DLDataTypeCode::FLOAT, 32) => Ok(DType::F32),
        (ffi::DLDataTypeCode::INT, 32) => Ok(DType::I32),
        (ffi::DLDataTypeCode::BFLOAT, 16) => Ok(DType::BF16),
        _ => UnsupportedDTypeSnafu { dtype }.fail(),
    }
}

fn import<M>(managed: Managed<M>) -> Result<Tensor>
where
    M: ManagedTensorBase + 'static,
{
    let (dtype, shape, len, data) = {
        let view = managed.validate().context(InvalidDescriptorSnafu)?;
        ensure!(
            view.device().device_type == ffi::DLDeviceType::CPU,
            UnsupportedDeviceSnafu {
                device_type: view.device().device_type,
            }
        );
        let dtype = dtype_from_dlpack(view.dtype())?;
        ensure!(
            view.is_compact().context(InvalidDescriptorSnafu)?,
            NonCompactSnafu
        );
        let shape = view.shape().to_vec();
        let len = view.num_bytes();
        // SAFETY: consuming a conforming DLPack producer transfers a readable
        // CPU allocation governed by the managed-tensor deleter.
        let pointer = unsafe { view.offset_bytes_ptr() }.context(InvalidDataSnafu)?;
        let data = NonNull::new(pointer.cast_mut()).unwrap_or_else(NonNull::dangling);
        (dtype, shape, len, data)
    };

    Tensor::builder(&shape, dtype)
        .context(InvalidTensorSnafu)?
        .from_storage(Storage::host(
            dtype,
            Owner {
                _managed: managed,
                data,
                len,
            },
        ))
        .context(InvalidTensorSnafu)
}

impl TryFrom<versioned::Dlpack> for Tensor {
    type Error = ImportError;

    fn try_from(value: versioned::Dlpack) -> Result<Self> {
        import(value)
    }
}

impl TryFrom<legacy::Dlpack> for Tensor {
    type Error = ImportError;

    fn try_from(value: legacy::Dlpack) -> Result<Self> {
        import(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlpark::{DlpackFlags, allocation::fixed, ffi::DLManagedTensorVersioned};
    use image::{ImageBuffer, Rgb};

    #[test]
    fn image_dlpack_import_is_zero_copy_and_keeps_shape() {
        let image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(2, 1, vec![0, 64, 128, 192, 255, 32])
            .unwrap();
        let source_pointer = image.as_raw().as_ptr().cast::<u8>();
        let mut initialized: fixed::Initialized<DLManagedTensorVersioned, 3> =
            Box::new(image).try_into().unwrap();
        initialized.set_flags(DlpackFlags::READ_ONLY).unwrap();
        // SAFETY: dlpark's image producer initialized every descriptor field
        // and transferred ownership of the boxed image into its deleter.
        let managed: versioned::Dlpack = unsafe { initialized.finish() };

        let tensor = Tensor::try_from(managed).unwrap();
        assert_eq!(tensor.shape(), [1, 2, 3]);
        assert_eq!(tensor.dtype(), DType::U8);
        assert_eq!(
            tensor.storage().unwrap().host_bytes().unwrap().as_ptr(),
            source_pointer
        );
        assert_eq!(
            tensor.host_view().unwrap().element(&[0, 1, 1]).unwrap(),
            [255]
        );
    }
}
