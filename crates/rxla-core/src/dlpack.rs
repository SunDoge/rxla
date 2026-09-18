//! Ownership-preserving DLPack interchange.
//!
//! Import retains the producer's managed tensor and therefore invokes its
//! deleter exactly once when the last rxla storage owner is dropped.

use crate::{Client, DType, Storage, Tensor};
use dlpark::{Managed, ManagedTensorBase, TryFromDlpack, ffi, legacy, versioned};
use snafu::{ResultExt, Snafu, ensure};
use std::ptr::NonNull;

/// DLPack import failures.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum ImportError {
    #[snafu(display("invalid DLPack tensor: {source}"))]
    InvalidDescriptor { source: dlpark::tensor::Error },
    #[snafu(display("unsupported DLPack device {device_type:?}"))]
    UnsupportedDevice { device_type: ffi::DLDeviceType },
    #[snafu(display(
        "DLPack device {device_type:?} is incompatible with PJRT platform {platform:?}"
    ))]
    DevicePlatformMismatch {
        device_type: ffi::DLDeviceType,
        platform: String,
    },
    #[snafu(display("invalid DLPack device id {device_id}"))]
    InvalidDeviceId { device_id: i32 },
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
    #[snafu(display("DLPack device import failed: {source}"))]
    DeviceImport { source: rxla_pjrt::Error },
    #[snafu(display("DLPack PJRT tensor construction failed: {source}"))]
    DeviceTensor { source: crate::Error },
}

type Result<T> = std::result::Result<T, ImportError>;

/// PJRT context required while consuming a DLPack tensor.
///
/// `S` is the producer-stream representation. Use [`PjrtImport::new`] when the
/// producer has already synchronized its device allocation, or
/// [`PjrtImport::with_stream`] with a backend stream adapter.
#[derive(Clone, Copy)]
pub struct PjrtImport<'a, S = ()> {
    client: &'a Client,
    stream: S,
}

impl<'a> PjrtImport<'a> {
    /// Create an import context without producer-stream synchronization.
    pub fn new(client: &'a Client) -> Self {
        Self { client, stream: () }
    }
}

impl<'a, S> PjrtImport<'a, S> {
    /// Create an import context carrying the producer's stream.
    pub fn with_stream(client: &'a Client, stream: S) -> Self {
        Self { client, stream }
    }
}

/// Converts a producer stream into the native handle expected by PJRT.
///
/// # Safety
///
/// Implementors must return either zero for already-ready data or a live
/// producer stream belonging to the DLPack tensor's backend and device. A
/// nonzero stream must remain valid through PJRT's external-buffer creation.
pub unsafe trait PjrtProducerStream {
    /// Return the backend-native producer stream handle.
    fn native_handle(&self) -> isize;
}

/// Non-owning backend-native stream handle for low-level integrations.
#[derive(Clone, Copy, Debug)]
pub struct NativeStream(isize);

impl NativeStream {
    /// Wrap a backend-native producer stream handle.
    ///
    /// # Safety
    ///
    /// `handle` must be zero for already-ready data or identify a live stream
    /// compatible with the imported tensor's backend and device.
    pub const unsafe fn from_raw(handle: isize) -> Self {
        Self(handle)
    }
}

// SAFETY: construction requires the caller to uphold the native stream
// validity, device, and backend contract.
unsafe impl PjrtProducerStream for NativeStream {
    fn native_handle(&self) -> isize {
        self.0
    }
}

struct Owner<M: ManagedTensorBase> {
    _managed: Managed<M>,
    data: NonNull<u8>,
    len: usize,
}

struct DeviceOwner<M: ManagedTensorBase> {
    _managed: Managed<M>,
}

// SAFETY: import consumes the managed tensor, validates CPU storage, and RXLA
// exposes the allocation only through immutable byte views. The DLPack deleter
// owns the transferred allocation and may run after the handle moves threads.
unsafe impl<M: ManagedTensorBase> Send for Owner<M> {}
unsafe impl<M: ManagedTensorBase> Sync for Owner<M> {}
// SAFETY: ownership has been consumed from the producer. PJRT invokes the
// callback after all device use, and the DLPack deleter is permitted to run on
// that callback thread.
unsafe impl<M: ManagedTensorBase> Send for DeviceOwner<M> {}

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

fn is_host_accessible(device_type: ffi::DLDeviceType) -> bool {
    matches!(
        device_type,
        ffi::DLDeviceType::CPU | ffi::DLDeviceType::CUDAHOST | ffi::DLDeviceType::ROCMHOST
    )
}

fn import_host<M>(managed: Managed<M>) -> Result<Tensor>
where
    M: ManagedTensorBase + 'static,
{
    let (dtype, shape, len, data) = {
        let view = managed.validate().context(InvalidDescriptorSnafu)?;
        ensure!(
            is_host_accessible(view.device().device_type),
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

/// Import CPU, pinned-host, CUDA, or CUDA-managed DLPack storage.
///
/// CPU-accessible allocations retain their producer directly as host storage.
/// CUDA allocations become zero-copy PJRT buffer views on the matching local
/// hardware id. `producer_stream` is the native CUDA stream containing work
/// needed to materialize the buffer; pass `None` only when it is already ready.
fn import_on_device<M>(
    managed: Managed<M>,
    client: &Client,
    producer_stream: Option<isize>,
) -> Result<Tensor>
where
    M: ManagedTensorBase + 'static,
{
    let device_type = managed
        .validate()
        .context(InvalidDescriptorSnafu)?
        .device()
        .device_type;
    if is_host_accessible(device_type) {
        return import_host(managed);
    }
    ensure!(
        matches!(
            device_type,
            ffi::DLDeviceType::CUDA | ffi::DLDeviceType::CUDAMANAGED
        ),
        UnsupportedDeviceSnafu { device_type }
    );
    let info = client.info().context(DeviceImportSnafu)?;
    ensure!(
        info.platform.to_ascii_lowercase().contains("cuda"),
        DevicePlatformMismatchSnafu {
            device_type,
            platform: info.platform,
        }
    );
    let (dtype, shape, device_id, pointer) = {
        let view = managed.validate().context(InvalidDescriptorSnafu)?;
        let dtype = dtype_from_dlpack(view.dtype())?;
        ensure!(
            view.is_compact().context(InvalidDescriptorSnafu)?,
            NonCompactSnafu
        );
        let device_id = view.device().device_id;
        ensure!(device_id >= 0, InvalidDeviceIdSnafu { device_id });
        // SAFETY: a conforming consumed DLPack tensor provides a device pointer
        // valid through its managed deleter; PJRT receives that owner below.
        let pointer = unsafe { view.offset_bytes_ptr() }.context(InvalidDataSnafu)?;
        (
            dtype,
            view.shape().to_vec(),
            device_id,
            pointer.cast_mut().cast(),
        )
    };
    let device_index = client
        .addressable_device_index(device_id)
        .context(DeviceImportSnafu)?;
    let buffer = unsafe {
        client.view_of_device_buffer(
            device_index,
            &shape,
            dtype,
            pointer,
            DeviceOwner { _managed: managed },
            producer_stream,
        )
    }
    .context(DeviceImportSnafu)?;
    Tensor::from_pjrt_buffer(buffer).context(DeviceTensorSnafu)
}

impl<'a, M> TryFromDlpack<Managed<M>, PjrtImport<'a>> for Tensor
where
    M: ManagedTensorBase + 'static,
{
    type Error = ImportError;

    unsafe fn try_from_dlpack(managed: Managed<M>, import: PjrtImport<'a>) -> Result<Self> {
        import_on_device(managed, import.client, None)
    }
}

impl<'a, M, S> TryFromDlpack<Managed<M>, PjrtImport<'a, S>> for Tensor
where
    M: ManagedTensorBase + 'static,
    S: PjrtProducerStream,
{
    type Error = ImportError;

    unsafe fn try_from_dlpack(managed: Managed<M>, import: PjrtImport<'a, S>) -> Result<Self> {
        import_on_device(managed, import.client, Some(import.stream.native_handle()))
    }
}

impl TryFrom<versioned::Dlpack> for Tensor {
    type Error = ImportError;

    fn try_from(value: versioned::Dlpack) -> Result<Self> {
        import_host(value)
    }
}

impl TryFrom<legacy::Dlpack> for Tensor {
    type Error = ImportError;

    fn try_from(value: legacy::Dlpack) -> Result<Self> {
        import_host(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlpark::{
        DlpackFlags,
        allocation::fixed,
        ffi::{DLDataType, DLDevice, DLManagedTensorVersioned},
        metadata::{Copied, Fixed},
    };
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

    #[test]
    #[ignore = "requires trusted PJRT_CUDA_PLUGIN_PATH"]
    fn cuda_dlpack_import_is_a_zero_copy_pjrt_view()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let client = unsafe { Client::load(std::env::var("PJRT_CUDA_PLUGIN_PATH")?) }?;
        let source = client.buffer(&[4], &[1.0_f32, 2.0, 3.0, 4.0])?;
        // SAFETY: `source` is retained as the DLPack manager context and the
        // descriptor is consumed by `import_on_device` before that owner moves.
        let pointer = unsafe { source.opaque_device_pointer()? };
        let prepared =
            Fixed::new(Copied([4_i64]), Copied([1_i64])).prepare::<DLManagedTensorVersioned>()?;
        let mut initialized = prepared.initialize(Box::new(source));
        initialized
            .set_data(pointer.as_ptr())
            .set_dtype(DLDataType::F32)
            .set_device(DLDevice::cuda(0));
        initialized.set_flags(DlpackFlags::READ_ONLY)?;
        // SAFETY: all descriptor fields are initialized and the boxed PJRT
        // buffer keeps the CUDA allocation alive until DLPack deletion.
        let managed: versioned::Dlpack = unsafe { initialized.finish() };

        // SAFETY: zero is PJRT's no-producer-stream sentinel and the source
        // upload completed before descriptor creation.
        let stream = unsafe { NativeStream::from_raw(0) };
        let tensor =
            unsafe { Tensor::try_from_dlpack(managed, PjrtImport::with_stream(&client, stream))? };
        assert_eq!(tensor.shape(), [4]);
        assert_eq!(tensor.dtype(), DType::F32);
        let imported = tensor.to_buffer(&client)?;
        // SAFETY: both pointers are compared only while `imported` retains the
        // external allocation; neither pointer is dereferenced here.
        assert_eq!(unsafe { imported.opaque_device_pointer()? }, pointer);
        assert_eq!(imported.to_vec::<f32>()?, [1., 2., 3., 4.]);
        Ok(())
    }
}
