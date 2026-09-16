//! Tensor descriptor and managed storage. No implicit evaluation or native views.
use super::*;
use half::{bf16, f16};
use rxla_pjrt::{BufferMemoryLayout, HostView, Shape, StridedLayout};
use smallvec::SmallVec;
use snafu::{OptionExt, ResultExt, Snafu, ensure};
use std::{
    cell::{OnceCell, RefCell},
    ops::Deref,
    ptr::NonNull,
    rc::{Rc, Weak},
};

struct LazySession {
    graph: Graph,
    inputs: RefCell<Vec<Tensor>>,
}

thread_local! {
    static LAZY_SESSION: RefCell<Weak<LazySession>> = const { RefCell::new(Weak::new()) };
}

fn current_lazy_session() -> Rc<LazySession> {
    LAZY_SESSION.with(|slot| {
        if let Some(session) = slot.borrow().upgrade() {
            return session;
        }
        let session = Rc::new(LazySession {
            graph: Graph::default(),
            inputs: RefCell::new(Vec::new()),
        });
        *slot.borrow_mut() = Rc::downgrade(&session);
        session
    })
}

fn lazy_session_for_graph(graph: &Graph) -> Option<Rc<LazySession>> {
    LAZY_SESSION.with(|slot| {
        slot.borrow()
            .upgrade()
            .filter(|session| std::sync::Arc::ptr_eq(&session.graph.0, &graph.0))
    })
}

/// Allocation ownership, independent of tensor shape, strides and offset.
/// Clones share the owner. Native allocations remain on their owning thread.
#[derive(Clone)]
pub struct Storage(Rc<StorageInner>);
enum StorageInner {
    Host {
        dtype: DType,
        owner: Box<dyn AsRef<[u8]>>,
    },
    Device(Rc<Buffer>),
}

struct TypedOwner<T: TensorElement>(Vec<T>);

impl<T: TensorElement> AsRef<[u8]> for TypedOwner<T> {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: TensorElement promises a plain byte-stable representation,
        // and the Vec keeps this allocation alive for the returned slice.
        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr().cast(),
                self.0.len() * std::mem::size_of::<T>(),
            )
        }
    }
}

struct BorrowedOwner {
    pointer: NonNull<u8>,
    len: usize,
}

struct RawOwner {
    pointer: NonNull<u8>,
    len: usize,
    deleter: Option<Box<dyn FnOnce()>>,
}

impl AsRef<[u8]> for RawOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: upheld by TensorBuilder::from_raw_parts's caller.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}

impl Drop for RawOwner {
    fn drop(&mut self) {
        if let Some(deleter) = self.deleter.take() {
            deleter();
        }
    }
}

impl AsRef<[u8]> for BorrowedOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: upheld by TensorBuilder::borrow_from_slice's caller for the
        // entire lifetime of every Tensor/Storage clone retaining this owner.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageKind {
    Host,
    Pjrt,
}

/// Failures while inspecting or transferring managed tensor storage.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum StorageError {
    #[snafu(display("dtype {dtype:?} has no supported host representation"))]
    UnsupportedStorageDType { dtype: DType },
    #[snafu(display(
        "layout element width {actual} does not match {dtype:?} storage width {expected}"
    ))]
    LayoutElementSize {
        dtype: DType,
        expected: usize,
        actual: usize,
    },
    #[snafu(display("operation requires host storage"))]
    ExpectedHostStorage,
    #[snafu(display("PJRT client has no selected device"))]
    NoSelectedDevice,
    #[snafu(display("host packing allocation failed: {source}"))]
    HostPackingAllocation {
        source: std::collections::TryReserveError,
    },
    #[snafu(transparent)]
    Pjrt { source: rxla_pjrt::Error },
}

pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// A scalar whose in-memory native-endian representation can be copied into or
/// retained as tensor host storage.
///
/// # Safety
///
/// Every bit pattern read from a value of this type must be valid to observe as
/// bytes, with no padding that may be uninitialized. `DTYPE.size_bytes()` must
/// equal `size_of::<Self>()`, and the representation must match PJRT's host
/// transfer representation for `DTYPE`.
pub unsafe trait TensorElement: Copy + 'static {
    const DTYPE: DType;
}

/// Failures specific to constructing host-backed tensors.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum TensorBuildError {
    #[snafu(display("invalid tensor shape: {source}"))]
    InvalidShape { source: rxla_pjrt::Error },
    #[snafu(display("dtype {dtype:?} has no supported host representation"))]
    UnsupportedDType { dtype: DType },
    #[snafu(display("declared dtype {declared:?} does not match element dtype {element:?}"))]
    DTypeMismatch { declared: DType, element: DType },
    #[snafu(display("shape {shape:?} requires {expected} elements, received {actual}"))]
    ElementCount {
        shape: Vec<i64>,
        expected: usize,
        actual: usize,
    },
    #[snafu(display(
        "TensorElement size {actual} does not match {dtype:?} element size {expected}"
    ))]
    ElementSize {
        dtype: DType,
        expected: usize,
        actual: usize,
    },
    #[snafu(display("tensor layout does not match the builder shape or dtype"))]
    LayoutMismatch,
    #[snafu(display("typed slice constructors require a dense layout"))]
    SliceRequiresDenseLayout,
    #[snafu(display("tensor byte length overflow"))]
    ByteLengthOverflow,
    #[snafu(display("tensor host allocation failed: {source}"))]
    HostAllocation {
        source: std::collections::TryReserveError,
    },
    #[snafu(display("invalid tensor host layout: {source}"))]
    InvalidLayout { source: rxla_pjrt::Error },
    #[snafu(display("invalid tensor host storage: {source}"))]
    InvalidStorage { source: crate::Error },
}

/// Failures while explicitly downloading a tensor from PJRT storage.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum TensorDownloadError {
    #[snafu(display("tensor has no materialized PJRT storage; evaluate it first"))]
    NotMaterialized,
    #[snafu(transparent)]
    Pjrt { source: rxla_pjrt::Error },
}

type TensorBuildResult<T> = std::result::Result<T, TensorBuildError>;

// SAFETY: these primitives have no padding and PJRT consumes native-endian host values.
unsafe impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;
}
unsafe impl TensorElement for u8 {
    const DTYPE: DType = DType::U8;
}
unsafe impl TensorElement for i32 {
    const DTYPE: DType = DType::I32;
}
unsafe impl TensorElement for f16 {
    const DTYPE: DType = DType::F16;
}
unsafe impl TensorElement for bf16 {
    const DTYPE: DType = DType::BF16;
}

/// Builder for a materialized host Tensor in the thread-local lazy program.
pub struct TensorBuilder {
    shape: Shape,
    dtype: DType,
    layout: Option<StridedLayout>,
}

impl TensorBuilder {
    /// Use a checked non-contiguous host layout instead of the default dense
    /// row-major layout. Its shape and element width must match this builder.
    pub fn layout(mut self, layout: StridedLayout) -> TensorBuildResult<Self> {
        ensure!(
            layout.shape() == &self.shape
                && self.dtype.size_bytes() == Some(layout.element_bytes()),
            LayoutMismatchSnafu
        );
        self.layout = Some(layout);
        Ok(self)
    }

    /// Copy typed values into dense native-endian host storage.
    pub fn copy_from_slice<T: TensorElement>(self, values: &[T]) -> TensorBuildResult<Tensor> {
        self.validate_elements::<T>(values.len())?;
        let byte_len = values
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .context(ByteLengthOverflowSnafu)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .context(HostAllocationSnafu)?;
        // SAFETY: TensorElement guarantees initialized, byte-stable storage.
        let source = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) };
        bytes.extend_from_slice(source);
        let dtype = self.dtype;
        self.from_storage(Storage::host(dtype, bytes))
    }

    /// Move a typed Vec into the Tensor without copying its payload.
    pub fn from_vec<T: TensorElement>(self, values: Vec<T>) -> TensorBuildResult<Tensor> {
        self.validate_elements::<T>(values.len())?;
        let dtype = self.dtype;
        self.from_storage(Storage::host(dtype, TypedOwner(values)))
    }

    /// Retain an existing immutable storage owner without copying its payload.
    pub fn from_storage(self, storage: Storage) -> TensorBuildResult<Tensor> {
        let element_bytes = self
            .dtype
            .size_bytes()
            .context(UnsupportedDTypeSnafu { dtype: self.dtype })?;
        let layout = match self.layout {
            Some(layout) => layout,
            None => {
                StridedLayout::row_major(self.shape, element_bytes).context(InvalidLayoutSnafu)?
            }
        };
        Tensor::from_host_storage(self.dtype, storage, layout).context(InvalidStorageSnafu)
    }

    /// Borrow a typed slice without copying or retaining its owner.
    ///
    /// # Safety
    ///
    /// The slice allocation must remain alive, readable, immutable, and at the
    /// same address until every Tensor, Storage, pending transfer, and lazy
    /// expression derived from the result has been dropped. The caller must
    /// also synchronize any external access to that allocation.
    pub unsafe fn borrow_from_slice<T: TensorElement>(
        self,
        values: &[T],
    ) -> TensorBuildResult<Tensor> {
        self.validate_elements::<T>(values.len())?;
        let len = values
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .context(ByteLengthOverflowSnafu)?;
        let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<u8>())
            .expect("slice pointers are nonnull, including empty slices");
        let dtype = self.dtype;
        self.from_storage(Storage::host(dtype, BorrowedOwner { pointer, len }))
    }

    /// Take ownership of a host allocation and its deleter without copying.
    ///
    /// The deleter is called exactly once, including when validation fails,
    /// after the last Tensor/Storage clone owning the allocation is dropped.
    ///
    /// # Safety
    ///
    /// `pointer..pointer + byte_len` must remain a readable, immutable host
    /// allocation until `deleter` is invoked. The deleter must release that
    /// allocation, must be safe to invoke on this thread, and must not unwind.
    pub unsafe fn from_raw_parts(
        self,
        pointer: NonNull<u8>,
        byte_len: usize,
        deleter: impl FnOnce() + 'static,
    ) -> TensorBuildResult<Tensor> {
        let dtype = self.dtype;
        self.from_storage(Storage::host(
            dtype,
            RawOwner {
                pointer,
                len: byte_len,
                deleter: Some(Box::new(deleter)),
            },
        ))
    }

    fn validate_elements<T: TensorElement>(&self, len: usize) -> TensorBuildResult<()> {
        ensure!(
            self.dtype == T::DTYPE,
            DTypeMismatchSnafu {
                declared: self.dtype,
                element: T::DTYPE,
            }
        );
        let expected_size = self
            .dtype
            .size_bytes()
            .context(UnsupportedDTypeSnafu { dtype: self.dtype })?;
        ensure!(
            expected_size == std::mem::size_of::<T>(),
            ElementSizeSnafu {
                dtype: self.dtype,
                expected: expected_size,
                actual: std::mem::size_of::<T>(),
            }
        );
        let expected = self.shape.numel().context(InvalidShapeSnafu)?;
        ensure!(
            expected == len,
            ElementCountSnafu {
                shape: self.shape.as_slice().to_vec(),
                expected,
                actual: len,
            }
        );
        ensure!(self.layout.is_none(), SliceRequiresDenseLayoutSnafu);
        Ok(())
    }
}

impl Storage {
    /// Retain an owned byte provider (Vec, Box, Arc, mmap wrapper, etc.).
    /// No payload copy occurs here. The owner must keep its logical contents
    /// stable while used as an input; no mutation is exposed by this API.
    /// Storage keeps the provider alive until its final owning handle is dropped.
    pub fn host(dtype: DType, owner: impl AsRef<[u8]> + 'static) -> Self {
        Self(Rc::new(StorageInner::Host {
            dtype,
            owner: Box::new(owner),
        }))
    }
    /// Retain a native allocation; no metadata query or payload copy.
    pub fn device(buffer: Rc<Buffer>) -> Self {
        Self(Rc::new(StorageInner::Device(buffer)))
    }
    pub fn kind(&self) -> StorageKind {
        match &*self.0 {
            StorageInner::Host { .. } => StorageKind::Host,
            StorageInner::Device(_) => StorageKind::Pjrt,
        }
    }
    /// Owner identity, not an overlap test for independently imported allocations.
    pub fn shares_owner_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub fn host_bytes(&self) -> Option<&[u8]> {
        match &*self.0 {
            StorageInner::Host { owner, .. } => Some(owner.as_ref().as_ref()),
            _ => None,
        }
    }
    pub fn buffer(&self) -> Option<&Rc<Buffer>> {
        match &*self.0 {
            StorageInner::Device(buffer) => Some(buffer),
            _ => None,
        }
    }
    /// Host metadata is explicit; native dtype comes from the opaque PJRT buffer.
    pub fn dtype(&self) -> StorageResult<DType> {
        match &*self.0 {
            StorageInner::Host { dtype, .. } => Ok(*dtype),
            StorageInner::Device(buffer) => Ok(buffer.dtype()?),
        }
    }
    /// Validate a raw-byte view against this storage's recorded dtype.
    pub fn view(&self, layout: StridedLayout) -> StorageResult<HostView<'_>> {
        let dtype = self.dtype()?;
        let element_bytes = dtype
            .size_bytes()
            .ok_or(StorageError::UnsupportedStorageDType { dtype })?;
        if layout.element_bytes() != element_bytes {
            return Err(StorageError::LayoutElementSize {
                dtype,
                expected: element_bytes,
                actual: layout.element_bytes(),
            });
        }
        Ok(HostView::new(
            self.host_bytes().ok_or(StorageError::ExpectedHostStorage)?,
            layout,
        )?)
    }
    /// Explicitly pack/upload host bytes in native endian order according to dtype.
    /// I32 and BF16 never pass through F32. Typed, aligned scratch exists only at
    /// this native transfer boundary, not in the retained storage representation.
    pub fn upload(&self, layout: &StridedLayout, client: &Client) -> StorageResult<Rc<Buffer>> {
        let selected = client
            .info()?
            .addressable_devices
            .iter()
            .position(|device| device.selected)
            .ok_or(StorageError::NoSelectedDevice)?;
        self.upload_on_device(layout, client, selected)
    }

    pub(crate) fn upload_on_device(
        &self,
        layout: &StridedLayout,
        client: &Client,
        device: usize,
    ) -> StorageResult<Rc<Buffer>> {
        let view = self.view(layout.clone())?;
        let shape = layout.shape().as_slice();
        let buffer = match self.dtype()? {
            DType::U8 => client.buffer_on_device(device, shape, &pack_values(&view, |b| b[0])?)?,
            DType::F16 => client.buffer_on_device(
                device,
                shape,
                &pack_values(&view, |b| {
                    f16::from_bits(u16::from_ne_bytes(b.try_into().unwrap()))
                })?,
            )?,
            DType::F32 => client.buffer_on_device(
                device,
                shape,
                &pack_values(&view, |b| f32::from_ne_bytes(b.try_into().unwrap()))?,
            )?,
            DType::I32 => client.buffer_on_device(
                device,
                shape,
                &pack_values(&view, |b| i32::from_ne_bytes(b.try_into().unwrap()))?,
            )?,
            DType::BF16 => client.buffer_on_device(
                device,
                shape,
                &pack_values(&view, |b| {
                    bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap()))
                })?,
            )?,
            dtype => return Err(StorageError::UnsupportedStorageDType { dtype }),
        };
        Ok(Rc::new(buffer))
    }
}

// Private boundary conversion only. Byte reads avoid unaligned typed references.
fn pack_values<T>(view: &HostView<'_>, decode: fn(&[u8]) -> T) -> StorageResult<Vec<T>> {
    let shape = view.layout().shape();
    let count = shape.numel()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|source| StorageError::HostPackingAllocation { source })?;
    let mut index: SmallVec<i64, 5> = std::iter::repeat_n(0, shape.rank()).collect();
    for _ in 0..count {
        values.push(decode(view.element(&index)?));
        for axis in (0..index.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape.as_slice()[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(values)
}

#[derive(Clone)]
enum Binding {
    Host {
        storage: Storage,
        layout: StridedLayout,
    },
    Device {
        storage: Storage,
        layout: BufferMemoryLayout,
    },
}

/// Layout metadata owned by a Tensor. Symbolic tensors have no physical layout;
/// host views retain checked striding and device tensors retain PJRT metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorLayout<'a> {
    Symbolic,
    Host(&'a StridedLayout),
    Device(&'a BufferMemoryLayout),
}

/// Opaque immutable implementation of a Tensor handle. Fields are private to the
/// crate; dereferencing a Tensor grants no mutation or access to native pointers.
/// Views/bindings own independent descriptors and may share a Storage owner.
#[doc(hidden)]
pub struct TensorDescriptor {
    trace: Option<TraceValue>,
    pub(crate) shape: SmallVec<i64, 5>,
    dtype: DType,
    binding: OnceCell<Binding>,
}

#[derive(Clone)]
struct TraceValue {
    graph: Graph,
    id: rxla_ir::SsaId,
    lazy: Option<Rc<LazySession>>,
}

impl Deref for Tensor {
    type Target = TensorDescriptor;
    fn deref(&self) -> &Self::Target {
        &self.descriptor
    }
}

impl Tensor {
    /// Whether both tensors belong to the same traced program.
    ///
    /// This checks provenance only; it does not compare shapes, values, or
    /// storage. Standalone tensors that have not joined a trace return false.
    pub fn same_trace(&self, other: &Self) -> bool {
        match (&self.trace, &other.trace) {
            (Some(left), Some(right)) => Arc::ptr_eq(&left.graph.0, &right.graph.0),
            _ => false,
        }
    }

    /// Whether both handles refer to the same SSA value in the same trace.
    /// This is identity, not elementwise equality.
    pub fn same_expression(&self, other: &Self) -> bool {
        self.same_trace(other)
            && self.trace.as_ref().map(|value| value.id)
                == other.trace.as_ref().map(|value| value.id)
    }

    /// Create an F32 scalar in the same trace as this tensor.
    pub fn scalar(&self, value: f32) -> Result<Self> {
        self.graph().constant(&[], &[value])
    }

    /// Create an I32 scalar in the same trace as this tensor.
    pub fn scalar_i32(&self, value: i32) -> Result<Self> {
        self.graph().scalar_i32(value)
    }

    /// Start construction of a host-backed tensor with explicit shape and dtype.
    pub fn builder(shape: impl AsRef<[i64]>, dtype: DType) -> TensorBuildResult<TensorBuilder> {
        let shape = Shape::new(shape.as_ref()).context(InvalidShapeSnafu)?;
        let element_bytes = dtype
            .size_bytes()
            .context(UnsupportedDTypeSnafu { dtype })?;
        let _ =
            StridedLayout::row_major(shape.clone(), element_bytes).context(InvalidLayoutSnafu)?;
        Ok(TensorBuilder {
            shape,
            dtype,
            layout: None,
        })
    }

    /// Copy a typed slice-like value into a host-backed tensor. The explicit
    /// dtype is checked against `T` rather than inferred from a function name.
    pub fn from_slice<T: TensorElement>(
        shape: impl AsRef<[i64]>,
        dtype: DType,
        values: impl AsRef<[T]>,
    ) -> TensorBuildResult<Self> {
        Self::builder(shape, dtype)?.copy_from_slice(values.as_ref())
    }

    /// Attach a logical sharding constraint without selecting physical devices.
    /// Reapplying the same constraint is idempotent; conflicting constraints on
    /// the same expression are rejected.
    pub fn with_sharding(&self, sharding: Sharding) -> Result<Self> {
        sharding.validate_shape(self.shape())?;
        self.graph().set_sharding(self.node_id(), sharding)?;
        Ok(self.clone())
    }

    pub fn sharding(&self) -> Result<Option<Sharding>> {
        self.graph().sharding(self.node_id())
    }

    pub(super) fn symbolic(graph: Graph, id: rxla_ir::SsaId, shape: &[i64], dtype: DType) -> Self {
        let lazy = lazy_session_for_graph(&graph);
        Self {
            descriptor: Rc::new(TensorDescriptor {
                trace: Some(TraceValue { graph, id, lazy }),
                shape: SmallVec::from_slice_copy(shape),
                dtype,
                binding: OnceCell::new(),
            }),
        }
    }
    /// Logical element type; reads metadata without querying storage or a device.
    pub fn dtype(&self) -> DType {
        self.descriptor.dtype
    }
    /// Create a materialized host leaf in the thread's private lazy session.
    /// The checked layout owns the logical shape; storage owns the payload
    /// without copying it.
    pub fn from_host_storage(
        dtype: DType,
        storage: Storage,
        layout: StridedLayout,
    ) -> Result<Self> {
        let actual = storage.dtype()?;
        if actual != dtype {
            return Err(Error::StorageDTypeMismatch {
                declared: dtype,
                actual,
            });
        }
        storage.view(layout.clone())?;
        let value = Self {
            descriptor: Rc::new(TensorDescriptor {
                trace: None,
                shape: SmallVec::from_slice_copy(layout.shape().as_slice()),
                dtype,
                binding: OnceCell::from(Binding::Host { storage, layout }),
            }),
        };
        value.into_lazy()
    }

    /// True when this value already owns host or device storage.
    pub fn is_materialized(&self) -> bool {
        self.binding.get().is_some()
    }
    pub(crate) fn is_implicit_lazy(&self) -> bool {
        self.trace
            .as_ref()
            .is_some_and(|trace| trace.lazy.is_some())
    }
    /// Layout known by this handle without another backend query.
    pub fn layout(&self) -> TensorLayout<'_> {
        match self.binding.get() {
            None => TensorLayout::Symbolic,
            Some(Binding::Host { layout, .. }) => TensorLayout::Host(layout),
            Some(Binding::Device { layout, .. }) => TensorLayout::Device(layout),
        }
    }
    /// Wrap an executor result as a materialized leaf in the implicit lazy graph.
    pub(crate) fn materialized(buffer: Buffer) -> Result<Self> {
        Self::from_device_buffer(Rc::new(buffer))?.into_lazy()
    }

    fn from_device_buffer(buffer: Rc<Buffer>) -> Result<Self> {
        let shape = buffer.dimensions()?;
        let dtype = buffer.dtype()?;
        let layout = buffer.memory_layout()?;
        Ok(Self {
            descriptor: Rc::new(TensorDescriptor {
                trace: None,
                shape: SmallVec::from_slice_copy(&shape),
                dtype,
                binding: OnceCell::from(Binding::Device {
                    storage: Storage::device(buffer),
                    layout,
                }),
            }),
        })
    }

    fn into_lazy(self) -> Result<Self> {
        if self.trace.is_some() {
            return Ok(self);
        }
        let session = current_lazy_session();
        let id = session.graph.parameter(TensorType {
            dims: self.shape().to_vec(),
            dtype: self.dtype(),
        })?;
        session.inputs.borrow_mut().push(self.clone());
        Ok(Self {
            descriptor: Rc::new(TensorDescriptor {
                trace: Some(TraceValue {
                    graph: session.graph.clone(),
                    id,
                    lazy: Some(session),
                }),
                shape: self.shape.clone(),
                dtype: self.dtype(),
                binding: self
                    .binding
                    .get()
                    .cloned()
                    .map_or_else(OnceCell::new, OnceCell::from),
            }),
        })
    }

    pub(crate) fn lazy_inputs(&self, parameters: &[usize]) -> Result<Vec<Tensor>> {
        let session = self
            .trace_value()?
            .lazy
            .as_ref()
            .ok_or(Error::ExplicitTraceEvaluation)?;
        let inputs = session.inputs.borrow();
        parameters
            .iter()
            .map(|&index| {
                inputs
                    .get(index)
                    .cloned()
                    .ok_or(Error::MissingLazyInput { index })
            })
            .collect()
    }

    pub(crate) fn materialize_all(outputs: &[Tensor], values: &[Tensor]) -> Result<()> {
        if outputs.len() != values.len() {
            return Err(Error::MaterializedOutputCount {
                expected: outputs.len(),
                actual: values.len(),
            });
        }
        let Some(first) = outputs.first() else {
            return Ok(());
        };
        let first_trace = first.trace_value()?;
        let first_session = first_trace
            .lazy
            .as_ref()
            .ok_or(Error::ExplicitTraceEvaluation)?;
        let bindings = outputs
            .iter()
            .zip(values)
            .enumerate()
            .map(|(index, (output, value))| {
                if output.shape() != value.shape() || output.dtype() != value.dtype() {
                    return Err(Error::MaterializedOutputMetadata { index });
                }
                value
                    .binding
                    .get()
                    .cloned()
                    .ok_or(Error::ExecutorOutputNotMaterialized { index })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut replacements = Vec::new();
        for (output, binding) in outputs.iter().zip(&bindings) {
            let trace = output.trace_value()?;
            let Some(session) = &trace.lazy else {
                return Err(Error::ExplicitTraceEvaluation);
            };
            if !std::rc::Rc::ptr_eq(session, first_session) {
                return Err(Error::LazySessionMismatch);
            }
            if replacements
                .iter()
                .any(|(id, _, _): &(rxla_ir::SsaId, TensorType, Tensor)| *id == trace.id)
            {
                continue;
            }
            let backing = Tensor {
                descriptor: Rc::new(TensorDescriptor {
                    trace: None,
                    shape: output.shape.clone(),
                    dtype: output.dtype(),
                    binding: OnceCell::from(binding.clone()),
                }),
            };
            replacements.push((
                trace.id,
                TensorType {
                    dims: output.shape().to_vec(),
                    dtype: output.dtype(),
                },
                backing,
            ));
        }

        let mut graph = first_session
            .graph
            .0
            .lock()
            .map_err(|_| Error::GraphLockPoisoned)?;
        let mut inputs = first_session.inputs.borrow_mut();
        for (id, ty, backing) in replacements {
            if graph.parameter_number(id)?.is_some() {
                continue;
            }
            let index = inputs.len();
            inputs.push(backing);
            graph.replace_with_parameter(id, index, &ty)?;
        }
        drop(inputs);
        drop(graph);

        for (output, binding) in outputs.iter().zip(bindings) {
            // Tensor handles are thread-affine. A pre-existing binding means an
            // alias or earlier eval already materialized this immutable value.
            let _ = output.binding.set(binding);
        }
        Ok(())
    }
    fn input_index(&self) -> Result<usize> {
        let trace = self.trace_value()?;
        let graph = self
            .trace_value()?
            .graph
            .0
            .lock()
            .map_err(|_| Error::GraphLockPoisoned)?;
        graph
            .parameter_number(trace.id)?
            .ok_or(Error::StorageBindingRequiresInput)
    }
    fn with_binding(&self, binding: Binding) -> Self {
        Self {
            descriptor: Rc::new(TensorDescriptor {
                trace: self.trace.clone(),
                shape: self.shape.clone(),
                dtype: self.dtype(),
                binding: OnceCell::from(binding),
            }),
        }
    }
    /// Attach managed host storage to an input. Does not change other clones,
    /// graph semantics or checkpoints. Strides/offset describe a read-only view.
    pub fn with_host_storage(&self, storage: Storage, layout: StridedLayout) -> Result<Self> {
        self.input_index()?;
        let actual_dtype = storage.dtype()?;
        if layout.shape().as_slice() != self.shape() || actual_dtype != self.dtype() {
            return Err(Error::HostInputMetadata {
                expected_shape: self.shape().to_vec(),
                expected_dtype: self.dtype(),
                actual_shape: layout.shape().as_slice().to_vec(),
                actual_dtype,
            });
        }
        storage.view(layout.clone())?;
        Ok(self.with_binding(Binding::Host { storage, layout }))
    }
    /// Attach a resident input without copying its data. Client identity is
    /// validated when used with an executor. This does not create a strided view.
    pub fn with_device_storage(&self, storage: Storage) -> Result<Self> {
        self.input_index()?;
        let buffer = storage.buffer().ok_or(Error::ExpectedDeviceStorage)?;
        let actual_dtype = buffer.dtype()?;
        let actual_shape = buffer.dimensions()?;
        if actual_dtype != self.dtype() || actual_shape != self.shape() {
            return Err(Error::DeviceInputMetadata {
                expected_shape: self.shape().to_vec(),
                expected_dtype: self.dtype(),
                actual_shape,
                actual_dtype,
            });
        }
        let layout = buffer.memory_layout()?;
        Ok(self.with_binding(Binding::Device { storage, layout }))
    }
    pub fn storage(&self) -> Option<&Storage> {
        self.binding.get().map(|b| match b {
            Binding::Host { storage, .. } | Binding::Device { storage, .. } => storage,
        })
    }

    /// Download a materialized PJRT tensor into a typed host vector.
    ///
    /// This never evaluates a lazy expression implicitly. Call [`Self::eval`]
    /// first, or use [`Runtime::eval`](crate::Runtime::eval), then select the
    /// Rust element type explicitly. Host-backed inputs remain available through
    /// their original owner or [`Storage::host_bytes`].
    pub fn to_vec<T: rxla_pjrt::Element>(
        &self,
    ) -> std::result::Result<Vec<T>, TensorDownloadError> {
        let buffer = self
            .storage()
            .and_then(Storage::buffer)
            .ok_or(TensorDownloadError::NotMaterialized)?;
        buffer
            .to_vec::<T>()
            .map_err(|source| TensorDownloadError::Pjrt { source })
    }

    /// Known host-view layout only. None means symbolic or native storage, not
    /// a dense default. Native layout is available through Buffer::memory_layout.
    pub fn host_layout(&self) -> Option<&StridedLayout> {
        match self.binding.get() {
            Some(Binding::Host { layout, .. }) => Some(layout),
            _ => None,
        }
    }
    pub fn host_view(&self) -> Result<HostView<'_>> {
        match self.binding.get() {
            Some(Binding::Host { storage, layout }) => Ok(storage.view(layout.clone())?),
            _ => Err(Error::MissingHostStorage),
        }
    }
    /// Explicitly upload/pack a managed host input, or retain its same-client
    /// device buffer. Native byte order is used; no implicit cross-client copy.
    /// Repeated host calls upload again: use to_device once for resident reuse.
    pub fn to_buffer(&self, client: &Client) -> Result<Rc<Buffer>> {
        match self.binding.get() {
            Some(Binding::Device { storage, .. }) => {
                let buffer = storage.buffer().ok_or(Error::ExpectedDeviceStorage)?;
                if !buffer.belongs_to(client) {
                    return Err(Error::ForeignClientStorage);
                }
                Ok(buffer.clone())
            }
            Some(Binding::Host { storage, layout }) => Ok(storage.upload(layout, client)?),
            None => Err(Error::MissingManagedStorage),
        }
    }

    pub(crate) fn to_buffer_on_device(&self, client: &Client, device: usize) -> Result<Rc<Buffer>> {
        match self.binding.get() {
            Some(Binding::Device { storage, .. }) => {
                let buffer = storage.buffer().ok_or(Error::ExpectedDeviceStorage)?;
                if !buffer.belongs_to(client) {
                    return Err(Error::ForeignClientStorage);
                }
                Ok(buffer.clone())
            }
            Some(Binding::Host { storage, layout }) => {
                Ok(storage.upload_on_device(layout, client, device)?)
            }
            None => Err(Error::MissingManagedStorage),
        }
    }
    /// Return a new input descriptor owning resident storage; original unchanged.
    pub fn to_device(&self, client: &Client) -> Result<Self> {
        if !self.is_implicit_lazy() {
            return self.with_device_storage(Storage::device(self.to_buffer(client)?));
        }
        if self.is_materialized() {
            if matches!(self.binding.get(), Some(Binding::Device { .. })) {
                self.to_buffer(client)?;
                return Ok(self.clone());
            }
            return Self::from_device_buffer(self.to_buffer(client)?)?.into_lazy();
        }
        self.with_device_storage(Storage::device(self.to_buffer(client)?))
    }

    fn trace_value(&self) -> Result<&TraceValue> {
        self.trace.as_ref().ok_or(Error::MissingExpression)
    }

    pub(crate) fn graph(&self) -> &Graph {
        &self.trace.as_ref().expect("tensor trace invariant").graph
    }

    pub(crate) fn node_id(&self) -> rxla_ir::SsaId {
        self.trace.as_ref().expect("tensor trace invariant").id
    }

    pub(crate) fn ty(&self) -> TensorType {
        TensorType {
            dims: self.shape().to_vec(),
            dtype: self.dtype(),
        }
    }
}

impl Compiler {
    /// Execute with managed input tensors in declared parameter order.
    /// Binding is explicit: graph nodes and LoweredProgram never capture storage.
    /// Resident inputs are reused; host inputs are packed/uploaded on each call.
    pub fn execute_bound(&mut self, output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Buffer>> {
        {
            let graph = output
                .graph()
                .0
                .lock()
                .map_err(|_| Error::GraphLockPoisoned)?;
            if graph.parameter_count()? != inputs.len() {
                return Err(err("managed input count mismatch"));
            }
        }
        for (index, input) in inputs.iter().enumerate() {
            if !Arc::ptr_eq(&output.graph().0, &input.graph().0)
                || input.input_index()? != index
                || input.storage().is_none()
            {
                return Err(err(
                    "managed inputs must match graph parameter order and own storage",
                ));
            }
        }
        let buffers = inputs
            .iter()
            .map(|input| input.to_buffer(&self.client))
            .collect::<Result<Vec<_>>>()?;
        let executable =
            self.compile_graph_outputs(output.graph(), std::slice::from_ref(output))?;
        executable.execute(&buffers.iter().map(|b| b.as_ref()).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_pjrt::{ByteStrides, Shape};
    use std::cell::Cell;

    struct Owner {
        bytes: Vec<u8>,
        drops: Rc<Cell<usize>>,
    }
    impl AsRef<[u8]> for Owner {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn one_pointer_handle_and_managed_view_lifetime() {
        assert_eq!(std::mem::size_of::<Tensor>(), std::mem::size_of::<usize>());
        let drops = Rc::new(Cell::new(0));
        let storage = Storage::host(
            DType::F32,
            Owner {
                bytes: (0..6).flat_map(|n| (n as f32).to_ne_bytes()).collect(),
                drops: drops.clone(),
            },
        );
        let graph = Graph::default();
        let input = graph.input(&[3, 2]).unwrap();
        let clone = input.clone();
        assert!(Rc::ptr_eq(&input.descriptor, &clone.descriptor));
        assert!(!input.shape.spilled());
        let layout = StridedLayout::new(
            Shape::new(&[3, 2]).unwrap(),
            ByteStrides::new(&[4, 12]),
            0,
            4,
        )
        .unwrap();
        let bound = input.with_host_storage(storage.clone(), layout).unwrap();
        let alias = bound.clone();
        assert!(input.storage().is_none());
        assert!(Rc::ptr_eq(&bound.descriptor, &alias.descriptor));
        assert!(bound.storage().unwrap().shares_owner_with(&storage));
        drop(bound);
        drop(storage);
        assert_eq!(drops.get(), 0);
        assert_eq!(
            alias.host_view().unwrap().element(&[2, 1]).unwrap(),
            5f32.to_ne_bytes()
        );
        drop(alias);
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn erased_storage_checks_dtype_even_when_width_matches() {
        let storage = Storage::host(DType::I32, vec![0u8; 8]);
        let layout = StridedLayout::row_major(Shape::new(&[2]).unwrap(), 4).unwrap();
        assert_eq!(storage.dtype().unwrap(), DType::I32);
        assert!(storage.view(layout.clone()).is_ok());
        let x = Graph::default().input(&[2]).unwrap();
        assert_eq!(x.dtype(), DType::F32);
        assert!(matches!(
            x.with_host_storage(storage, layout),
            Err(Error::HostInputMetadata {
                expected_shape,
                expected_dtype: DType::F32,
                actual_shape,
                actual_dtype: DType::I32,
            }) if expected_shape == [2] && actual_shape == [2]
        ));
        let bf16 = Storage::host(DType::BF16, vec![0u8; 8]);
        assert!(matches!(
            bf16.view(StridedLayout::row_major(Shape::new(&[2]).unwrap(), 4).unwrap()),
            Err(StorageError::LayoutElementSize {
                dtype: DType::BF16,
                expected: 2,
                actual: 4,
            })
        ));

        let unknown = DType::from_raw(99_999);
        assert!(matches!(
            Storage::host(unknown, Vec::<u8>::new()).view(
                StridedLayout::row_major(Shape::new(&[0]).unwrap(), 1).unwrap()
            ),
            Err(StorageError::UnsupportedStorageDType { dtype }) if dtype == unknown
        ));
    }

    #[test]
    fn tensor_storage_binding_failures_are_structured() {
        let shape = Shape::new(&[1]).unwrap();
        let layout = StridedLayout::row_major(shape, 4).unwrap();
        assert!(matches!(
            Tensor::from_host_storage(
                DType::F32,
                Storage::host(DType::I32, 1i32.to_ne_bytes()),
                layout.clone(),
            ),
            Err(Error::StorageDTypeMismatch {
                declared: DType::F32,
                actual: DType::I32,
            })
        ));

        let graph = Graph::default();
        let computed = graph.constant(&[1], &[1.0]).unwrap().exp().unwrap();
        assert!(matches!(
            computed.with_host_storage(Storage::host(DType::F32, 1f32.to_ne_bytes()), layout,),
            Err(Error::StorageBindingRequiresInput)
        ));
        assert!(matches!(
            computed.host_view(),
            Err(Error::MissingHostStorage)
        ));
    }

    #[test]
    fn standalone_host_tensors_are_materialized_lazy_leaves() {
        let integer = Tensor::from_slice([2], DType::I32, [-1, i32::MAX]).unwrap();
        assert!(integer.is_materialized());
        assert_eq!(integer.dtype(), DType::I32);
        assert_eq!(integer.shape(), [2]);
        assert!(matches!(integer.layout(), TensorLayout::Host(_)));
        assert_eq!(integer.storage().unwrap().kind(), StorageKind::Host);
        assert_eq!(
            integer.host_view().unwrap().element(&[1]).unwrap(),
            i32::MAX.to_ne_bytes()
        );
        assert!(!integer.wrapping_add_scalar(1).unwrap().is_materialized());

        let bf16_values = [bf16::from_f32(1.0), bf16::from_bits(0x8000)];
        let bf16 = Tensor::from_slice([2], DType::BF16, bf16_values).unwrap();
        assert_eq!(bf16.dtype(), DType::BF16);
        assert_eq!(
            bf16.host_view().unwrap().element(&[0]).unwrap(),
            0x3f80u16.to_ne_bytes()
        );
        assert!(Tensor::from_slice([2], DType::F32, [1.0]).is_err());
    }

    #[test]
    fn typed_download_requires_explicit_device_materialization() {
        let host = Tensor::from_slice([2], DType::F32, [1.0, 2.0]).unwrap();
        let lazy = host.add(&host).unwrap();

        assert!(matches!(
            host.to_vec::<f32>(),
            Err(TensorDownloadError::NotMaterialized)
        ));
        assert!(matches!(
            lazy.to_vec::<f32>(),
            Err(TensorDownloadError::NotMaterialized)
        ));
        assert!(!lazy.is_materialized());
    }

    #[test]
    fn materialization_rejects_explicit_traces_without_panicking() {
        assert!(Tensor::materialize_all(&[], &[]).is_ok());
        let value = Tensor::from_slice([1], DType::F32, [1.0]).unwrap();
        assert!(matches!(
            Tensor::materialize_all(&[], std::slice::from_ref(&value)),
            Err(Error::MaterializedOutputCount {
                expected: 0,
                actual: 1
            })
        ));

        let explicit = Graph::default().input(&[1]).unwrap();
        assert!(matches!(
            Tensor::materialize_all(&[explicit], std::slice::from_ref(&value)),
            Err(Error::ExplicitTraceEvaluation)
        ));

        let lazy = value.add_scalar(1.0).unwrap();
        let wrong_shape = Tensor::from_slice([2], DType::F32, [1.0, 2.0]).unwrap();
        assert!(matches!(
            Tensor::materialize_all(
                std::slice::from_ref(&lazy),
                std::slice::from_ref(&wrong_shape)
            ),
            Err(Error::MaterializedOutputMetadata { index: 0 })
        ));
        let unmaterialized = value.add_scalar(2.0).unwrap();
        assert!(matches!(
            Tensor::materialize_all(
                std::slice::from_ref(&lazy),
                std::slice::from_ref(&unmaterialized)
            ),
            Err(Error::ExecutorOutputNotMaterialized { index: 0 })
        ));
    }

    #[test]
    fn tensor_builder_distinguishes_invalid_dtype_and_element_count() {
        let mismatch = Tensor::builder([2], DType::I32)
            .unwrap()
            .copy_from_slice(&[1.0_f32, 2.0])
            .err()
            .unwrap();
        assert!(matches!(
            mismatch,
            TensorBuildError::DTypeMismatch {
                declared: DType::I32,
                element: DType::F32
            }
        ));

        let count = Tensor::from_slice([2, 2], DType::F32, [1.0_f32, 2.0])
            .err()
            .unwrap();
        assert!(matches!(
            count,
            TensorBuildError::ElementCount {
                expected: 4,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn tensor_builder_owned_and_borrowed_paths_preserve_allocations() {
        let owned = vec![1_i32, 2, 3];
        let owned_pointer = owned.as_ptr().cast::<u8>();
        let tensor = Tensor::builder([3], DType::I32)
            .unwrap()
            .from_vec(owned)
            .unwrap();
        assert_eq!(
            tensor.storage().unwrap().host_bytes().unwrap().as_ptr(),
            owned_pointer
        );

        let borrowed = [4_i32, 5, 6];
        let borrowed_pointer = borrowed.as_ptr().cast::<u8>();
        // SAFETY: `borrowed` remains immutable and alive until `view` is dropped.
        let view = unsafe {
            Tensor::builder([3], DType::I32)
                .unwrap()
                .borrow_from_slice(&borrowed)
                .unwrap()
        };
        assert_eq!(
            view.storage().unwrap().host_bytes().unwrap().as_ptr(),
            borrowed_pointer
        );
        assert_eq!(
            view.host_view().unwrap().element(&[2]).unwrap(),
            6_i32.to_ne_bytes()
        );
    }

    #[test]
    fn tensor_builder_raw_owner_invokes_deleter_once_after_last_clone() {
        let drops = Rc::new(Cell::new(0));
        let allocation = vec![1.0_f32, 2.0];
        let pointer = NonNull::new(allocation.as_ptr().cast_mut().cast::<u8>()).unwrap();
        let observed = drops.clone();
        // SAFETY: the deleter closure owns `allocation`, keeping the pointer
        // readable and stable until rxla releases the final storage owner.
        let tensor = unsafe {
            Tensor::builder([2], DType::F32)
                .unwrap()
                .from_raw_parts(pointer, 8, move || {
                    drop(allocation);
                    observed.set(observed.get() + 1);
                })
                .unwrap()
        };
        let alias = tensor.clone();
        drop(tensor);
        assert_eq!(drops.get(), 0);
        drop(alias);
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn tensor_builder_supports_native_f16_and_rejects_slice_layouts() {
        let values = [f16::from_f32(1.5), f16::from_f32(-2.0)];
        let tensor = Tensor::from_slice([2], DType::F16, values).unwrap();
        assert_eq!(tensor.dtype(), DType::F16);
        assert_eq!(
            tensor.host_view().unwrap().element(&[0]).unwrap(),
            values[0].to_bits().to_ne_bytes()
        );

        let layout = StridedLayout::row_major(Shape::new(&[2]).unwrap(), 4).unwrap();
        let error = Tensor::builder([2], DType::F32)
            .unwrap()
            .layout(layout)
            .unwrap()
            .copy_from_slice(&[1.0, 2.0])
            .err()
            .unwrap();
        assert!(matches!(error, TensorBuildError::SliceRequiresDenseLayout));
    }

    #[test]
    fn materialization_is_shared_and_cuts_the_evaluated_graph() {
        let x = Tensor::from_slice([2], DType::F32, [1.0, 2.0]).unwrap();
        let y = Tensor::from_slice([2], DType::F32, [3.0, 4.0]).unwrap();
        let output = x.add(&y).unwrap();
        let alias = output.clone();
        let value = Tensor::from_slice([2], DType::F32, [4.0, 6.0]).unwrap();

        Tensor::materialize_all(std::slice::from_ref(&output), &[value]).unwrap();

        assert!(output.is_materialized());
        assert!(alias.is_materialized());
        let graph = output.graph().0.lock().unwrap();
        let ir = &*graph;
        assert!(ir.parameter_number(output.node_id()).unwrap().is_some());
        assert!(ir.operand_ids(output.node_id()).unwrap().is_empty());
        drop(graph);

        let continued = output.add(&x).unwrap();
        let graph = continued.graph().0.lock().unwrap();
        let ir = &*graph;
        let operands = ir.operand_ids(continued.node_id()).unwrap();
        assert_eq!(operands[0], output.node_id());
        assert!(ir.parameter_number(operands[0]).unwrap().is_some());
    }

    #[test]
    fn byte_storage_packing_preserves_integer_and_bf16_bits() {
        let words = [i32::MIN, 16_777_217, i32::MAX];
        let bytes: Vec<_> = std::iter::once(0u8)
            .chain(words.into_iter().flat_map(i32::to_ne_bytes))
            .collect();
        let storage = Storage::host(DType::I32, bytes);
        let layout =
            StridedLayout::new(Shape::new(&[3]).unwrap(), ByteStrides::new(&[-4]), 9, 4).unwrap();
        let view = storage.view(layout).unwrap();
        let values = pack_values(&view, |b| i32::from_ne_bytes(b.try_into().unwrap())).unwrap();
        assert_eq!(values, [i32::MAX, 16_777_217, i32::MIN]);
        let bits = [0x8000u16, 0x7fc1, 0x3f80];
        let storage = Storage::host(
            DType::BF16,
            bits.into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<_>>(),
        );
        let view = storage
            .view(StridedLayout::row_major(Shape::new(&[3]).unwrap(), 2).unwrap())
            .unwrap();
        assert_eq!(
            pack_values(&view, |b| u16::from_ne_bytes(b.try_into().unwrap())).unwrap(),
            bits
        );
    }

    #[test]
    fn bindings_do_not_change_graph_or_prepared_abi() {
        let graph = Graph::default();
        let x = graph.input(&[2]).unwrap();
        let y = x.mul(&x).unwrap();
        let before = graph.stablehlo(&y).unwrap();
        let layout = StridedLayout::row_major(Shape::new(&[2]).unwrap(), 4).unwrap();
        let storage = Storage::host(DType::F32, vec![0u8; 8]);
        let bound = x
            .with_host_storage(storage.clone(), layout.clone())
            .unwrap();
        assert!(
            y.with_host_storage(storage.clone(), layout.clone())
                .is_err()
        );
        assert!(
            x.with_host_storage(Storage::host(DType::F32, vec![0u8; 7]), layout)
                .is_err()
        );
        assert!(
            x.with_host_storage(
                storage,
                StridedLayout::row_major(Shape::new(&[2]).unwrap(), 2).unwrap()
            )
            .is_err()
        );
        assert_eq!(before, graph.stablehlo(&y).unwrap());
        assert_eq!(
            y.sum(&[0], false).unwrap().grad(&[bound]).unwrap()[0].shape(),
            [2]
        );
        assert!(graph.input(&[1; 6]).unwrap().shape.spilled());
    }
}
