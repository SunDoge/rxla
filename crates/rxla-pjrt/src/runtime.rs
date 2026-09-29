#![allow(unused_unsafe)] // Macro guards remain safe when invoked outside unsafe blocks.
use crate::distributed::{DistributedClientConfig, KeyValueCallbackState};
use crate::sys::*;
use libloading::Library;
use snafu::{OptionExt, ResultExt, Snafu, ensure};
use std::{
    collections::BTreeMap,
    mem::ManuallyDrop,
    path::Path,
    ptr::{self, NonNull},
    sync::Arc,
};

/// Borrowed compiler input passed to PJRT together with its format label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Program<'a> {
    format: &'a str,
    code: &'a [u8],
}

impl<'a> Program<'a> {
    /// Construct a program in a plugin-defined format.
    pub const fn new(format: &'a str, code: &'a [u8]) -> Self {
        Self { format, code }
    }

    /// XLA `HloModuleProto` bytes.
    pub const fn hlo(code: &'a [u8]) -> Self {
        Self::new("hlo", code)
    }

    /// StableHLO/MHLO MLIR text or bytecode accepted by the selected plugin.
    pub const fn mlir(code: &'a [u8]) -> Self {
        Self::new("mlir", code)
    }

    pub const fn format(self) -> &'a str {
        self.format
    }

    pub const fn code(self) -> &'a [u8] {
        self.code
    }
}

/// Stable classification of error codes returned by a PJRT plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PjrtErrorCode {
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    Unrecognized(u32),
}

#[allow(non_upper_case_globals)]
impl From<PJRT_Error_Code> for PjrtErrorCode {
    fn from(code: PJRT_Error_Code) -> Self {
        match code {
            PJRT_Error_Code_PJRT_Error_Code_CANCELLED => Self::Cancelled,
            PJRT_Error_Code_PJRT_Error_Code_UNKNOWN => Self::Unknown,
            PJRT_Error_Code_PJRT_Error_Code_INVALID_ARGUMENT => Self::InvalidArgument,
            PJRT_Error_Code_PJRT_Error_Code_DEADLINE_EXCEEDED => Self::DeadlineExceeded,
            PJRT_Error_Code_PJRT_Error_Code_NOT_FOUND => Self::NotFound,
            PJRT_Error_Code_PJRT_Error_Code_ALREADY_EXISTS => Self::AlreadyExists,
            PJRT_Error_Code_PJRT_Error_Code_PERMISSION_DENIED => Self::PermissionDenied,
            PJRT_Error_Code_PJRT_Error_Code_RESOURCE_EXHAUSTED => Self::ResourceExhausted,
            PJRT_Error_Code_PJRT_Error_Code_FAILED_PRECONDITION => Self::FailedPrecondition,
            PJRT_Error_Code_PJRT_Error_Code_ABORTED => Self::Aborted,
            PJRT_Error_Code_PJRT_Error_Code_OUT_OF_RANGE => Self::OutOfRange,
            PJRT_Error_Code_PJRT_Error_Code_UNIMPLEMENTED => Self::Unimplemented,
            PJRT_Error_Code_PJRT_Error_Code_INTERNAL => Self::Internal,
            PJRT_Error_Code_PJRT_Error_Code_UNAVAILABLE => Self::Unavailable,
            PJRT_Error_Code_PJRT_Error_Code_DATA_LOSS => Self::DataLoss,
            PJRT_Error_Code_PJRT_Error_Code_UNAUTHENTICATED => Self::Unauthenticated,
            other => Self::Unrecognized(other),
        }
    }
}

impl PjrtErrorCode {
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::Cancelled
                | Self::DeadlineExceeded
                | Self::ResourceExhausted
                | Self::Aborted
                | Self::Unavailable
        )
    }
}

#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum Error {
    #[snafu(display("invalid argument: {message}"))]
    InvalidArgument { message: String },
    #[snafu(display("invalid state: {message}"))]
    InvalidState { message: String },
    #[snafu(display("incompatible PJRT plugin: {message}"))]
    IncompatiblePlugin { message: String },
    #[snafu(display("invalid PJRT response: {message}"))]
    InvalidPluginData { message: String },
    #[snafu(display("unsupported PJRT operation: {message}"))]
    Unsupported { message: String },
    #[snafu(display("resource limit exceeded: {message}"))]
    LimitExceeded { message: String },
    #[snafu(display("PJRT plugin error {code:?}: {message}"))]
    Pjrt {
        code: PjrtErrorCode,
        message: String,
    },
    #[snafu(display("load PJRT plugin: {source}"))]
    LoadPlugin { source: libloading::Error },
    #[snafu(display("load native support library: {source}"))]
    LoadSupportLibrary { source: libloading::Error },
    #[snafu(display("resolve PJRT entry point: {source}"))]
    ResolveEntryPoint { source: libloading::Error },
    #[snafu(display("{operation}: {source}"))]
    ResourceExhausted {
        operation: &'static str,
        source: std::collections::TryReserveError,
    },
    #[snafu(display("invalid PJRT text for {operation}: {source}"))]
    InvalidPluginEncoding {
        operation: &'static str,
        source: std::str::Utf8Error,
    },
}

impl Error {
    /// Return the backend status code when this error originated in PJRT.
    pub fn pjrt_code(&self) -> Option<PjrtErrorCode> {
        match self {
            Self::Pjrt { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Whether retrying may succeed without changing the request.
    pub fn is_retryable(&self) -> bool {
        self.pjrt_code().is_some_and(PjrtErrorCode::is_retryable)
    }
}
type Result<T> = std::result::Result<T, Error>;

// Read only a function slot proven to exist, never dereference a whole newer
// Rust API struct against a shorter plugin table.
macro_rules! function {
    ($api:expr, $name:ident) => {{
        let api = $api;
        let offset = std::mem::offset_of!(PJRT_Api, $name);
        ensure!(
            unsafe { (*api).struct_size } >= offset + std::mem::size_of::<usize>(),
            IncompatiblePluginSnafu {
                message: concat!("missing API slot: ", stringify!($name)),
            }
        );
        unsafe { ptr::addr_of!((*api).$name).read() }.context(IncompatiblePluginSnafu {
            message: concat!("null API slot: ", stringify!($name)),
        })?
    }};
}
macro_rules! args {
    ($ty:ident, $size:ident) => {{
        // All C fields are integers, nullable pointers/function pointers, or bool.
        let mut a: $ty = unsafe { std::mem::zeroed() };
        a.struct_size = $size as usize;
        a
    }};
}

// Construct a versioned argument block, populate its input fields, resolve the
// checked API slot, and translate the returned PJRT error in one expression.
// Keeping this private makes the safe Rust API explicit while centralizing the
// easy-to-get-wrong C ABI ceremony.
macro_rules! pjrt_call {
    ($plugin:expr, $name:ident, $ty:ident, $size:ident $(, $field:ident = $value:expr)* $(,)?) => {{
        let plugin = $plugin;
        let mut a = args!($ty, $size);
        $(a.$field = $value;)*
        plugin.check(unsafe { function!(plugin.api(), $name)(&mut a) })?;
        a
    }};
}

#[path = "metadata.rs"]
mod metadata;
pub use metadata::{ClientInfo, DeviceInfo, MemoryInfo};
#[path = "memory_stats.rs"]
mod memory_stats;
pub use memory_stats::DeviceMemoryStats;
#[path = "layout.rs"]
mod layout;
pub use layout::BufferMemoryLayout;
#[path = "host_layout.rs"]
mod host_layout;
pub use host_layout::{ByteStrides, HostView, Shape, StridedLayout};
#[path = "client_options.rs"]
mod client_options;
pub use client_options::{ClientOptionValue, ClientOptions};

struct PluginInner {
    api: *const PJRT_Api,
    // PJRT has no process-wide shutdown/join contract. Client destruction does
    // not prove all plugin TLS callbacks or background threads have terminated.
    // Retain the loader reference for process lifetime, including failed loads.
    _library: Option<ManuallyDrop<Library>>,
}

/// An initialized PJRT implementation, independent of any client it creates.
///
/// Cloning this handle does not reload or reinitialize the implementation.
/// Dynamically loaded libraries remain mapped until process exit because PJRT
/// does not define a process-wide shutdown contract.
#[derive(Clone)]
pub struct Plugin(Arc<PluginInner>);

/// A native library whose symbols are globally visible to compiled XLA code.
///
/// The mapping is intentionally retained for the process lifetime: compiled
/// executables can keep unresolved host-symbol references after this Rust
/// value is dropped. This is useful for small runtime libraries accompanying
/// offline-generated custom kernels, such as Mosaic GPU artifacts.
pub struct NativeSupportLibrary {
    _library: ManuallyDrop<Library>,
}

unsafe impl Send for NativeSupportLibrary {}
unsafe impl Sync for NativeSupportLibrary {}

impl NativeSupportLibrary {
    /// Load a trusted library with process-global symbol visibility.
    ///
    /// # Safety
    /// The library and all of its initialization code must be trusted, and its
    /// exported ABI must match every compiled artifact that references it.
    #[cfg(unix)]
    pub unsafe fn load_global(path: impl AsRef<Path>) -> Result<Self> {
        use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};

        let library = unsafe { UnixLibrary::open(Some(path.as_ref()), RTLD_NOW | RTLD_GLOBAL) }
            .map(Library::from)
            .context(LoadSupportLibrarySnafu)?;
        Ok(Self {
            _library: ManuallyDrop::new(library),
        })
    }
}

/// Calling convention used by a GPU custom-call handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuCustomCallApi {
    /// XLA's legacy untyped custom-call ABI.
    Legacy = 0,
    /// The typed XLA FFI ABI.
    TypedFfi = 1,
}

/// Process-lifetime entry points for one GPU custom call.
///
/// XLA FFI permits optional lifecycle handlers and requires an execute handler.
/// The code containing these pointers must remain loaded for the lifetime of
/// the PJRT plugin.
#[derive(Clone, Copy, Debug)]
pub struct GpuCustomCallHandlers {
    pub instantiate: Option<NonNull<std::ffi::c_void>>,
    pub prepare: Option<NonNull<std::ffi::c_void>>,
    pub initialize: Option<NonNull<std::ffi::c_void>>,
    pub execute: NonNull<std::ffi::c_void>,
}

impl GpuCustomCallHandlers {
    pub const fn execute_only(execute: NonNull<std::ffi::c_void>) -> Self {
        Self {
            instantiate: None,
            prepare: None,
            initialize: None,
            execute,
        }
    }
}

#[repr(C)]
struct PjrtGpuRegisterCustomCallArgs {
    struct_size: usize,
    function_name: *const std::ffi::c_char,
    function_name_size: usize,
    api_version: i32,
    handler_instantiate: *mut std::ffi::c_void,
    handler_prepare: *mut std::ffi::c_void,
    handler_initialize: *mut std::ffi::c_void,
    handler_execute: *mut std::ffi::c_void,
}

type PjrtGpuRegisterCustomCall =
    unsafe extern "C" fn(*mut PjrtGpuRegisterCustomCallArgs) -> *mut PJRT_Error;

#[repr(C)]
struct PjrtGpuCustomCallExtension {
    base: PJRT_Extension_Base,
    custom_call: Option<PjrtGpuRegisterCustomCall>,
}

// PJRT API tables are immutable after initialization and implementations expose
// thread-safe client, buffer, and executable interfaces. The dynamic library is
// retained for process lifetime, so sharing this pointer cannot race unloading.
unsafe impl Send for PluginInner {}
unsafe impl Sync for PluginInner {}

// Resolve the cleanup and synchronization capabilities before any operation can
// return owned resources or start a transfer borrowing Rust host memory. Other
// capabilities (e.g. serialization) remain optional and are checked on demand.
fn validate_lifetime_api(api: *const PJRT_Api) -> Result<()> {
    let _ = function!(api, PJRT_Error_Message);
    let _ = function!(api, PJRT_Error_GetCode);
    let _ = function!(api, PJRT_Error_Destroy);
    let _ = function!(api, PJRT_Event_Await);
    let _ = function!(api, PJRT_Event_Destroy);
    let _ = function!(api, PJRT_Client_Destroy);
    let _ = function!(api, PJRT_Buffer_Destroy);
    let _ = function!(api, PJRT_LoadedExecutable_Destroy);
    let _ = function!(api, PJRT_Executable_Destroy);
    Ok(())
}
impl Plugin {
    fn api(&self) -> *const PJRT_Api {
        self.0.api
    }

    fn extension(&self, kind: PJRT_Extension_Type) -> Option<NonNull<PJRT_Extension_Base>> {
        let mut extension = unsafe { (*self.api()).extension_start };
        let mut visited = std::collections::HashSet::new();
        while let Some(current) = NonNull::new(extension) {
            if !visited.insert(current.as_ptr() as usize) {
                return None;
            }
            let base = unsafe { current.as_ptr().read() };
            if base.struct_size < PJRT_Extension_Base_STRUCT_SIZE as usize {
                return None;
            }
            if base.type_ == kind {
                return Some(current);
            }
            extension = base.next;
        }
        None
    }

    /// Whether this implementation advertises the experimental direct
    /// collectives extension. This is distinct from graph collectives: a CUDA
    /// plugin can execute StableHLO collectives through NCCL without exposing
    /// this extension.
    pub fn supports_collectives_extension(&self) -> bool {
        self.extension(PJRT_Extension_Type_PJRT_Extension_Type_Collectives)
            .is_some()
    }

    /// Whether this plugin can register CUDA/ROCm custom-call handlers.
    pub fn supports_gpu_custom_calls(&self) -> bool {
        let Some(extension) =
            self.extension(PJRT_Extension_Type_PJRT_Extension_Type_Gpu_Custom_Call)
        else {
            return false;
        };
        let extension = extension.cast::<PjrtGpuCustomCallExtension>();
        let required = std::mem::offset_of!(PjrtGpuCustomCallExtension, custom_call)
            + std::mem::size_of::<Option<PjrtGpuRegisterCustomCall>>();
        unsafe {
            (*extension.as_ptr()).base.struct_size >= required
                && (*extension.as_ptr()).custom_call.is_some()
        }
    }

    /// Register a process-lifetime GPU custom-call implementation.
    ///
    /// # Safety
    /// Every handler must use the selected XLA ABI and remain executable for
    /// the process lifetime. Its implementation must uphold the buffer, stream,
    /// aliasing and asynchronous-lifetime contracts described by XLA FFI.
    pub unsafe fn register_gpu_custom_call(
        &self,
        name: &str,
        api: GpuCustomCallApi,
        handlers: GpuCustomCallHandlers,
    ) -> Result<()> {
        let extension = self
            .extension(PJRT_Extension_Type_PJRT_Extension_Type_Gpu_Custom_Call)
            .context(UnsupportedSnafu {
                message: "PJRT plugin does not expose GPU custom-call registration",
            })?
            .cast::<PjrtGpuCustomCallExtension>();
        let required = std::mem::offset_of!(PjrtGpuCustomCallExtension, custom_call)
            + std::mem::size_of::<Option<PjrtGpuRegisterCustomCall>>();
        ensure!(
            unsafe { (*extension.as_ptr()).base.struct_size } >= required,
            IncompatiblePluginSnafu {
                message: "GPU custom-call extension is shorter than its required prefix",
            }
        );
        let register =
            unsafe { (*extension.as_ptr()).custom_call }.context(IncompatiblePluginSnafu {
                message: "GPU custom-call extension has a null registration function",
            })?;
        let mut args = PjrtGpuRegisterCustomCallArgs {
            struct_size: std::mem::size_of::<PjrtGpuRegisterCustomCallArgs>(),
            function_name: name.as_ptr().cast(),
            function_name_size: name.len(),
            api_version: api as i32,
            handler_instantiate: handlers
                .instantiate
                .map_or(ptr::null_mut(), NonNull::as_ptr),
            handler_prepare: handlers.prepare.map_or(ptr::null_mut(), NonNull::as_ptr),
            handler_initialize: handlers.initialize.map_or(ptr::null_mut(), NonNull::as_ptr),
            handler_execute: handlers.execute.as_ptr(),
        };
        self.check(unsafe { register(&mut args) })
    }

    /// Whether the PJRT API table advertises both DMA map/unmap entry points.
    ///
    /// A backend may still return `Unimplemented`; callers normally should use
    /// `Client::upload_pinned`, which falls back transparently in that case.
    pub fn has_dma_mapping_api(&self) -> bool {
        let api = self.api();
        for offset in [
            std::mem::offset_of!(PJRT_Api, PJRT_Client_DmaMap),
            std::mem::offset_of!(PJRT_Api, PJRT_Client_DmaUnmap),
        ] {
            if unsafe { (*api).struct_size } < offset + std::mem::size_of::<usize>() {
                return false;
            }
        }
        unsafe {
            ptr::addr_of!((*api).PJRT_Client_DmaMap).read().is_some()
                && ptr::addr_of!((*api).PJRT_Client_DmaUnmap).read().is_some()
        }
    }

    /// Load and initialize a trusted PJRT dynamic library.
    ///
    /// A path containing no directory separators may be resolved by the
    /// platform dynamic loader (for example through `LD_LIBRARY_PATH`).
    ///
    /// # Safety
    /// The library must be trusted and implement the PJRT ABI correctly.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let library =
            ManuallyDrop::new(unsafe { Library::new(path.as_ref()) }.context(LoadPluginSnafu)?);
        let get =
            unsafe { library.get::<unsafe extern "C" fn() -> *const PJRT_Api>(b"GetPjrtApi\0") }
                .context(ResolveEntryPointSnafu)?;
        let api = unsafe { get() };
        unsafe { Self::from_parts(api, Some(library)) }
    }

    /// Initialize a PJRT implementation already linked into the process.
    ///
    /// This is the static-link counterpart to [`Self::load`].
    ///
    /// # Safety
    /// `api` must remain valid until process exit and every function pointer it
    /// exposes must obey the PJRT ABI. Calling this more than once for an
    /// implementation that does not permit repeated initialization is invalid.
    pub unsafe fn from_api(api: *const PJRT_Api) -> Result<Self> {
        unsafe { Self::from_parts(api, None) }
    }

    unsafe fn from_parts(
        api: *const PJRT_Api,
        library: Option<ManuallyDrop<Library>>,
    ) -> Result<Self> {
        ensure!(
            !api.is_null(),
            InvalidPluginDataSnafu {
                message: "null PJRT API",
            }
        );
        let prefix = std::mem::offset_of!(PJRT_Api, PJRT_Error_Destroy);
        ensure!(
            unsafe { (*api).struct_size } >= prefix,
            IncompatiblePluginSnafu {
                message: "truncated PJRT API",
            }
        );
        let version = unsafe { ptr::addr_of!((*api).pjrt_api_version).read() };
        ensure!(
            version.major_version == PJRT_API_MAJOR as i32,
            IncompatiblePluginSnafu {
                message: "incompatible PJRT major version",
            }
        );
        validate_lifetime_api(api)?;
        let plugin = Self(Arc::new(PluginInner {
            api,
            _library: library,
        }));
        let mut init = args!(
            PJRT_Plugin_Initialize_Args,
            PJRT_Plugin_Initialize_Args_STRUCT_SIZE
        );
        plugin.check(unsafe { function!(api, PJRT_Plugin_Initialize)(&mut init) })?;
        Ok(plugin)
    }

    /// Create a client using this initialized implementation.
    pub fn create_client(&self, options: &ClientOptions) -> Result<Client> {
        self.create_client_on_device(options, 0)
    }

    /// Create one node of a distributed client using a process-shared
    /// rendezvous store. The configuration owns the store for the client
    /// lifetime and injects the standard GPU `node_id`/`num_nodes` options.
    pub fn create_distributed_client(
        &self,
        options: &ClientOptions,
        config: DistributedClientConfig,
    ) -> Result<Client> {
        let options = config.options(options);
        options.validate()?;
        Client::from_plugin_with_callbacks(self.clone(), &options, 0, Some(config.callbacks()))
    }

    /// Create a client selecting one addressable device as its default.
    pub fn create_client_on_device(
        &self,
        options: &ClientOptions,
        addressable_device_index: usize,
    ) -> Result<Client> {
        Client::from_plugin(self.clone(), options, addressable_device_index)
    }

    fn check(&self, err: *mut PJRT_Error) -> Result<()> {
        if err.is_null() {
            return Ok(());
        }
        let mut status = args!(PJRT_Error_GetCode_Args, PJRT_Error_GetCode_Args_STRUCT_SIZE);
        status.error = err;
        let status_error = unsafe { function!(self.api(), PJRT_Error_GetCode)(&mut status) };
        let code = if status_error.is_null() {
            status.code.into()
        } else {
            let mut destroy = args!(PJRT_Error_Destroy_Args, PJRT_Error_Destroy_Args_STRUCT_SIZE);
            destroy.error = status_error;
            unsafe { function!(self.api(), PJRT_Error_Destroy)(&mut destroy) };
            PjrtErrorCode::Unknown
        };
        let mut msg = args!(PJRT_Error_Message_Args, PJRT_Error_Message_Args_STRUCT_SIZE);
        msg.error = err;
        unsafe { function!(self.api(), PJRT_Error_Message)(&mut msg) };
        let message = if msg.message.is_null() {
            "unspecified plugin error".into()
        } else {
            String::from_utf8_lossy(unsafe {
                std::slice::from_raw_parts(msg.message.cast(), msg.message_size)
            })
            .into_owned()
        };
        let mut destroy = args!(PJRT_Error_Destroy_Args, PJRT_Error_Destroy_Args_STRUCT_SIZE);
        destroy.error = err;
        unsafe { function!(self.api(), PJRT_Error_Destroy)(&mut destroy) };
        Err(Error::Pjrt { code, message })
    }
    fn wait(&self, event: *mut PJRT_Event) -> Result<()> {
        ensure!(
            !event.is_null(),
            InvalidPluginDataSnafu {
                message: "plugin returned a null completion event",
            }
        );
        let mut a = args!(PJRT_Event_Await_Args, PJRT_Event_Await_Args_STRUCT_SIZE);
        a.event = event;
        let result = self.check(unsafe { function!(self.api(), PJRT_Event_Await)(&mut a) });
        let mut d = args!(PJRT_Event_Destroy_Args, PJRT_Event_Destroy_Args_STRUCT_SIZE);
        d.event = event;
        let cleanup = self.check(unsafe { function!(self.api(), PJRT_Event_Destroy)(&mut d) });
        result.and(cleanup)
    }
}

/// Explicit, process-local collection of named PJRT implementations.
///
/// The registry never reads environment variables or scans library paths.
/// Applications decide how names and plugin sources are configured.
#[derive(Default)]
pub struct PluginRegistry {
    plugins: BTreeMap<String, Plugin>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load and register a dynamic PJRT implementation.
    ///
    /// # Safety
    /// The resolved library must be trusted and implement the PJRT ABI.
    pub unsafe fn register_dynamic(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<&mut Self> {
        let name = validate_plugin_name(name.into(), &self.plugins)?;
        let plugin = unsafe { Plugin::load(path) }?;
        self.plugins.insert(name, plugin);
        Ok(self)
    }

    /// Register a PJRT API already linked into the process.
    ///
    /// # Safety
    /// The pointer and implementation must satisfy [`Plugin::from_api`].
    pub unsafe fn register_api(
        &mut self,
        name: impl Into<String>,
        api: *const PJRT_Api,
    ) -> Result<&mut Self> {
        let name = validate_plugin_name(name.into(), &self.plugins)?;
        let plugin = unsafe { Plugin::from_api(api) }?;
        self.plugins.insert(name, plugin);
        Ok(self)
    }

    pub fn plugin(&self, name: &str) -> Result<&Plugin> {
        self.plugins.get(name).context(InvalidArgumentSnafu {
            message: format!("PJRT plugin {name:?} is not registered"),
        })
    }

    pub fn create_client(&self, name: &str, options: &ClientOptions) -> Result<Client> {
        self.plugin(name)?.create_client(options)
    }

    pub fn create_distributed_client(
        &self,
        name: &str,
        options: &ClientOptions,
        config: DistributedClientConfig,
    ) -> Result<Client> {
        self.plugin(name)?
            .create_distributed_client(options, config)
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.plugins.keys().map(String::as_str)
    }
}

fn validate_plugin_name(name: String, plugins: &BTreeMap<String, Plugin>) -> Result<String> {
    ensure!(
        !name.is_empty(),
        InvalidArgumentSnafu {
            message: "PJRT plugin name must be nonempty",
        }
    );
    ensure!(
        !plugins.contains_key(&name),
        InvalidStateSnafu {
            message: format!("PJRT plugin {name:?} is already registered"),
        }
    );
    Ok(name)
}

struct ClientInner {
    plugin: Plugin,
    raw: *mut PJRT_Client,
    device: *mut PJRT_Device,
    addressable_devices: Vec<*mut PJRT_Device>,
    // PJRT may use rendezvous callbacks from background threads for the full
    // client lifetime. Keep their stable allocation alive until after destroy.
    _key_value_callbacks: Option<Box<KeyValueCallbackState>>,
}
impl ClientInner {
    fn destroy(&self) -> Result<()> {
        let mut a = args!(
            PJRT_Client_Destroy_Args,
            PJRT_Client_Destroy_Args_STRUCT_SIZE
        );
        a.client = self.raw;
        self.plugin
            .check(unsafe { function!(self.plugin.api(), PJRT_Client_Destroy)(&mut a) })
    }
}
impl Drop for ClientInner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

/// Thread-safe shared PJRT client handle.
#[derive(Clone)]
pub struct Client(Arc<ClientInner>);

// PJRT clients are designed for concurrent use. Native device pointers are
// borrowed from the client and remain valid until the last Arc destroys it.
unsafe impl Send for ClientInner {}
unsafe impl Sync for ClientInner {}

unsafe extern "C" fn drop_external_device_owner<O: Send + 'static>(
    _device_buffer_ptr: *mut std::ffi::c_void,
    user_arg: *mut std::ffi::c_void,
) {
    if !user_arg.is_null() {
        drop(unsafe { Box::from_raw(user_arg.cast::<O>()) });
    }
}

impl Client {
    pub fn addressable_device_count(&self) -> usize {
        self.0.addressable_devices.len()
    }

    /// Resolve a backend-local hardware ordinal (for example a DLPack CUDA
    /// device id) to this client's addressable-device index.
    pub fn addressable_device_index(&self, local_hardware_id: i32) -> Result<usize> {
        for (index, &device) in self.0.addressable_devices.iter().enumerate() {
            let mut args = args!(
                PJRT_Device_LocalHardwareId_Args,
                PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE
            );
            args.device = device;
            self.0.plugin.check(unsafe {
                function!(self.0.plugin.api(), PJRT_Device_LocalHardwareId)(&mut args)
            })?;
            if args.local_hardware_id == local_hardware_id {
                return Ok(index);
            }
        }
        InvalidArgumentSnafu {
            message: format!(
                "local hardware device {local_hardware_id} is not addressable by this client"
            ),
        }
        .fail()
    }

    pub fn supports_collectives_extension(&self) -> bool {
        self.0.plugin.supports_collectives_extension()
    }

    pub fn has_dma_mapping_api(&self) -> bool {
        self.0.plugin.has_dma_mapping_api()
    }

    /// Load a trusted native plugin. It executes native code in this process.
    /// Successfully opened libraries are retained until process exit, even if
    /// later API validation/initialization fails. Clients and device resources
    /// are still destroyed normally. Plugin hot-unloading is not supported.
    ///
    /// # Safety
    /// The library must implement the PJRT ABI correctly and be trusted.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        unsafe { Self::load_with_options(path, &ClientOptions::new()) }
    }

    /// Load a trusted plugin with explicit, plugin-specific client options.
    /// Unknown names or unsupported types are reported by the plugin. Empty
    /// options preserve plugin defaults; no environment variables are translated.
    /// Empty names are rejected before loading native code. Repeated calls to
    /// [`ClientOptions::set`] replace the earlier value with that name.
    /// Libraries have the same process lifetime as with [`Self::load`].
    ///
    /// # Safety
    /// The library must implement the PJRT ABI correctly and be trusted,
    /// including consuming/copying option data before client creation returns.
    pub unsafe fn load_with_options(
        path: impl AsRef<Path>,
        options: &ClientOptions,
    ) -> Result<Self> {
        unsafe { Self::load_on_device(path, options, 0) }
    }

    /// Load a trusted plugin and create one node of a distributed PJRT client.
    ///
    /// CUDA PJRT uses the supplied rendezvous callbacks to exchange bootstrap
    /// information for cross-process collective communication (normally NCCL).
    /// Client creation may block until every configured node joins.
    ///
    /// # Safety
    /// The native library must be trusted and implement the PJRT ABI correctly.
    pub unsafe fn load_distributed(
        path: impl AsRef<Path>,
        options: &ClientOptions,
        config: DistributedClientConfig,
    ) -> Result<Self> {
        options.validate()?;
        let plugin = unsafe { Plugin::load(path) }?;
        plugin.create_distributed_client(options, config)
    }

    /// Create a client whose default uploads and single-device executables use
    /// the selected index in the plugin's addressable-device list. The index is
    /// not a global device ID or CUDA ordinal. Selection is fixed for the
    /// lifetime of this client and all its clones; per-device upload methods and
    /// sharded executables still use this client's other addressable devices.
    ///
    /// Raw HLO compilation options must explicitly target the selected device
    /// (see [`Self::info`] for its global ID). Wrong-device executables, including
    /// deserialized ones, are rejected. Multi-device executables require
    /// [`Executable::execute_sharded`]. This does not enable native cross-client
    /// device copies. Invalid indices destroy the newly created client before
    /// returning an error; plugin libraries remain loaded.
    ///
    /// # Safety
    /// The plugin and option lifetime contract are the same as
    /// [`Self::load_with_options`].
    pub unsafe fn load_on_device(
        path: impl AsRef<Path>,
        options: &ClientOptions,
        addressable_device_index: usize,
    ) -> Result<Self> {
        // Preserve local validation before executing any native library code.
        options.validate()?;
        let plugin = unsafe { Plugin::load(path) }?;
        plugin.create_client_on_device(options, addressable_device_index)
    }

    fn from_plugin(
        plugin: Plugin,
        options: &ClientOptions,
        addressable_device_index: usize,
    ) -> Result<Self> {
        Self::from_plugin_with_callbacks(plugin, options, addressable_device_index, None)
    }

    fn from_plugin_with_callbacks(
        plugin: Plugin,
        options: &ClientOptions,
        addressable_device_index: usize,
        mut key_value_callbacks: Option<Box<KeyValueCallbackState>>,
    ) -> Result<Self> {
        let raw_options = options.encode()?;
        let api = plugin.api();
        let mut create = args!(PJRT_Client_Create_Args, PJRT_Client_Create_Args_STRUCT_SIZE);
        create.create_options = if raw_options.is_empty() {
            ptr::null()
        } else {
            raw_options.as_ptr()
        };
        create.num_options = raw_options.len();
        if let Some(callbacks) = key_value_callbacks.as_mut() {
            callbacks.install(&mut create);
        }
        plugin.check(unsafe { function!(api, PJRT_Client_Create)(&mut create) })?;
        ensure!(
            !create.client.is_null(),
            InvalidPluginDataSnafu {
                message: "null client",
            }
        );
        let mut inner = ClientInner {
            plugin,
            raw: create.client,
            device: ptr::null_mut(),
            addressable_devices: Vec::new(),
            _key_value_callbacks: key_value_callbacks,
        };
        let mut devices = args!(
            PJRT_Client_AddressableDevices_Args,
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE
        );
        devices.client = inner.raw;
        inner
            .plugin
            .check(unsafe { function!(api, PJRT_Client_AddressableDevices)(&mut devices) })?;
        ensure!(
            devices.num_addressable_devices != 0,
            InvalidPluginDataSnafu {
                message: "no addressable device",
            }
        );
        if addressable_device_index >= devices.num_addressable_devices {
            return InvalidArgumentSnafu {
                message: format!(
                    "addressable device index {addressable_device_index} out of range for {} devices",
                    devices.num_addressable_devices
                ),
            }
            .fail();
        }
        ensure!(
            !devices.addressable_devices.is_null(),
            InvalidPluginDataSnafu {
                message: "null addressable device list",
            }
        );
        inner.addressable_devices = unsafe {
            std::slice::from_raw_parts(devices.addressable_devices, devices.num_addressable_devices)
        }
        .to_vec();
        ensure!(
            !inner
                .addressable_devices
                .iter()
                .any(|device| device.is_null()),
            InvalidPluginDataSnafu {
                message: "null addressable device",
            }
        );
        inner.device = inner.addressable_devices[addressable_device_index];
        Ok(Self(Arc::new(inner)))
    }

    pub fn buffer<T: Element>(&self, dims: &[i64], data: &[T]) -> Result<Buffer> {
        let device_index = self
            .0
            .addressable_devices
            .iter()
            .position(|&device| device == self.0.device)
            .expect("selected device belongs to addressable devices");
        self.buffer_on_device(device_index, dims, data)
    }

    /// Upload a host tensor directly into an explicit PJRT memory space.
    ///
    /// This avoids staging a checkpoint in device memory before moving it to
    /// pinned-host offload storage. Whether a particular memory kind accepts
    /// host uploads is defined by the backend.
    pub fn buffer_in_memory<T: Element>(
        &self,
        memory: &Memory,
        dims: &[i64],
        data: &[T],
    ) -> Result<Buffer> {
        ensure!(
            Arc::ptr_eq(&self.0, &memory.client),
            InvalidArgumentSnafu {
                message: "memory belongs to a different PJRT client",
            }
        );
        let count = dims
            .iter()
            .try_fold(1usize, |count, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| count.checked_mul(dimension))
            })
            .context(InvalidArgumentSnafu {
                message: "invalid or overflowing shape",
            })?;
        ensure!(
            count == data.len(),
            InvalidArgumentSnafu {
                message: "shape/data length mismatch",
            }
        );
        let mut args = args!(
            PJRT_Client_BufferFromHostBuffer_Args,
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
        );
        args.client = self.0.raw;
        args.memory = memory.raw.as_ptr();
        args.data = data.as_ptr().cast();
        args.type_ = T::DTYPE.as_raw();
        args.dims = dims.as_ptr();
        args.num_dims = dims.len();
        args.host_buffer_semantics =
            PJRT_HostBufferSemantics_PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        self.0.plugin.check(unsafe {
            function!(self.0.plugin.api(), PJRT_Client_BufferFromHostBuffer)(&mut args)
        })?;
        let buffer = Buffer {
            inner: Arc::new(BufferInner::new(self.0.clone(), args.buffer)?),
        };
        self.0.plugin.wait(args.done_with_host_buffer)?;
        Ok(buffer)
    }

    /// Create a PJRT view over an externally owned device allocation.
    ///
    /// PJRT receives `owner` and drops it through the on-delete callback only
    /// after the returned buffer no longer uses the allocation. No data copy is
    /// performed. A null stream means the producer has already made the data
    /// ready; otherwise it is the backend-native producer stream handle.
    ///
    /// # Safety
    ///
    /// `device_buffer_ptr` must address a compact row-major allocation of
    /// `dtype` and `dims` on `device_index`. The allocation and any metadata
    /// captured by `owner` must remain valid until `owner` is dropped. `stream`
    /// must be a valid stream for the selected backend when non-null.
    pub unsafe fn view_of_device_buffer<O: Send + 'static>(
        &self,
        device_index: usize,
        dims: &[i64],
        dtype: DType,
        device_buffer_ptr: *mut std::ffi::c_void,
        owner: O,
        stream: Option<isize>,
    ) -> Result<Buffer> {
        let device =
            *self
                .0
                .addressable_devices
                .get(device_index)
                .context(InvalidArgumentSnafu {
                    message: "addressable device index out of range",
                })?;
        let elements = dims.iter().try_fold(1usize, |count, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| count.checked_mul(dimension))
        });
        ensure!(
            elements.is_some(),
            InvalidArgumentSnafu {
                message: "invalid or overflowing shape",
            }
        );
        ensure!(
            elements == Some(0) || !device_buffer_ptr.is_null(),
            InvalidArgumentSnafu {
                message: "non-empty external device buffer has a null pointer",
            }
        );
        let create = function!(self.0.plugin.api(), PJRT_Client_CreateViewOfDeviceBuffer);
        let owner = Box::into_raw(Box::new(owner));
        let mut args = args!(
            PJRT_Client_CreateViewOfDeviceBuffer_Args,
            PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE
        );
        args.client = self.0.raw;
        args.device_buffer_ptr = device_buffer_ptr;
        args.dims = dims.as_ptr();
        args.num_dims = dims.len();
        args.element_type = dtype.as_raw();
        args.device = device;
        args.on_delete_callback = Some(drop_external_device_owner::<O>);
        args.on_delete_callback_arg = owner.cast();
        args.stream = stream.unwrap_or_default();
        if let Err(error) = self.0.plugin.check(unsafe { create(&mut args) }) {
            drop(unsafe { Box::from_raw(owner) });
            return Err(error);
        }
        let inner = match BufferInner::new(self.0.clone(), args.buffer) {
            Ok(inner) => inner,
            Err(error) => {
                drop(unsafe { Box::from_raw(owner) });
                return Err(error);
            }
        };
        Ok(Buffer {
            inner: Arc::new(inner),
        })
    }

    /// Register owned host memory for DMA and start an asynchronous upload.
    ///
    /// The returned handle owns both the allocation and its native DMA mapping
    /// until PJRT reports that the host bytes are no longer needed. Plugins
    /// DMA mapping is used when the backend implements it. Missing API slots or
    /// an `Unimplemented` response fall back to retaining ordinary owned memory
    /// until the host-transfer event completes.
    pub fn upload_pinned<T: Element>(
        &self,
        dims: &[i64],
        data: Vec<T>,
    ) -> Result<PendingHostUpload<T>> {
        let device_index = self
            .0
            .addressable_devices
            .iter()
            .position(|&device| device == self.0.device)
            .expect("selected device belongs to addressable devices");
        self.upload_pinned_on_device(device_index, dims, data)
    }

    pub fn upload_pinned_on_device<T: Element>(
        &self,
        device_index: usize,
        dims: &[i64],
        mut data: Vec<T>,
    ) -> Result<PendingHostUpload<T>> {
        let count = dims
            .iter()
            .try_fold(1usize, |count, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| count.checked_mul(dimension))
            })
            .context(InvalidArgumentSnafu {
                message: "invalid or overflowing shape",
            })?;
        ensure!(
            count == data.len(),
            InvalidArgumentSnafu {
                message: "shape/data length mismatch",
            }
        );
        let device =
            *self
                .0
                .addressable_devices
                .get(device_index)
                .context(InvalidArgumentSnafu {
                    message: "addressable device index out of range",
                })?;
        let bytes = std::mem::size_of_val(data.as_slice());
        let mut mapped = false;
        if bytes != 0 && self.has_dma_mapping_api() {
            let mut map = args!(PJRT_Client_DmaMap_Args, PJRT_Client_DmaMap_Args_STRUCT_SIZE);
            map.client = self.0.raw;
            map.data = data.as_mut_ptr().cast();
            map.size = bytes;
            let result = self
                .0
                .plugin
                .check(unsafe { function!(self.0.plugin.api(), PJRT_Client_DmaMap)(&mut map) });
            match result {
                Ok(()) => mapped = true,
                Err(error) if error.pjrt_code() == Some(PjrtErrorCode::Unimplemented) => {}
                Err(error) => return Err(error),
            }
        }
        let mut upload = args!(
            PJRT_Client_BufferFromHostBuffer_Args,
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
        );
        upload.client = self.0.raw;
        upload.device = device;
        upload.data = data.as_ptr().cast();
        upload.type_ = T::DTYPE.as_raw();
        upload.dims = dims.as_ptr();
        upload.num_dims = dims.len();
        upload.host_buffer_semantics =
            PJRT_HostBufferSemantics_PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        if let Err(error) = self.0.plugin.check(unsafe {
            function!(self.0.plugin.api(), PJRT_Client_BufferFromHostBuffer)(&mut upload)
        }) {
            if mapped {
                let _ = dma_unmap(&self.0, data.as_mut_ptr().cast());
            }
            return Err(error);
        }
        if upload.done_with_host_buffer.is_null() {
            if mapped {
                let _ = dma_unmap(&self.0, data.as_mut_ptr().cast());
            }
            return InvalidPluginDataSnafu {
                message: "plugin returned a null host upload completion event",
            }
            .fail();
        }
        let inner = match BufferInner::new(self.0.clone(), upload.buffer) {
            Ok(inner) => inner,
            Err(error) => {
                let _ = self.0.plugin.wait(upload.done_with_host_buffer);
                if mapped {
                    let _ = dma_unmap(&self.0, data.as_mut_ptr().cast());
                }
                return Err(error);
            }
        };
        let buffer = Buffer {
            inner: Arc::new(inner),
        };
        Ok(PendingHostUpload {
            client: self.0.clone(),
            data: Some(data),
            buffer: Some(buffer),
            event: Some(upload.done_with_host_buffer),
            mapped,
        })
    }
    pub fn buffer_on_device<T: Element>(
        &self,
        device_index: usize,
        dims: &[i64],
        data: &[T],
    ) -> Result<Buffer> {
        let _span = tracing::trace_span!("pjrt.host_to_device", rank = dims.len(), elements = data.len(), dtype = ?T::DTYPE).entered();
        let device =
            *self
                .0
                .addressable_devices
                .get(device_index)
                .context(InvalidArgumentSnafu {
                    message: "addressable device index out of range",
                })?;
        let count = dims
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .context(InvalidArgumentSnafu {
                message: "invalid or overflowing shape",
            })?;
        ensure!(
            count == data.len(),
            InvalidArgumentSnafu {
                message: "shape/data length mismatch",
            }
        );
        let mut a = args!(
            PJRT_Client_BufferFromHostBuffer_Args,
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
        );
        a.client = self.0.raw;
        a.device = device;
        a.data = data.as_ptr().cast();
        a.type_ = T::DTYPE.as_raw();
        a.dims = dims.as_ptr();
        a.num_dims = dims.len();
        a.host_buffer_semantics =
            PJRT_HostBufferSemantics_PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        self.0.plugin.check(unsafe {
            function!(self.0.plugin.api(), PJRT_Client_BufferFromHostBuffer)(&mut a)
        })?;
        let buffer = Buffer {
            inner: Arc::new(BufferInner::new(self.0.clone(), a.buffer)?),
        };
        self.0.plugin.wait(a.done_with_host_buffer)?;
        Ok(buffer)
    }

    pub fn compile(&self, source: Program<'_>, options: &[u8]) -> Result<Executable> {
        let _span = tracing::debug_span!(
            "pjrt.compile",
            format = source.format(),
            program_bytes = source.code().len(),
            options_bytes = options.len()
        )
        .entered();
        ensure!(
            !source.format().is_empty(),
            InvalidArgumentSnafu {
                message: "program format must not be empty",
            }
        );
        let mut program = args!(PJRT_Program, PJRT_Program_STRUCT_SIZE);
        program.code = source.code().as_ptr().cast_mut().cast();
        program.code_size = source.code().len();
        program.format = source.format().as_ptr().cast();
        program.format_size = source.format().len();
        let mut a = args!(
            PJRT_Client_Compile_Args,
            PJRT_Client_Compile_Args_STRUCT_SIZE
        );
        a.client = self.0.raw;
        a.program = &program;
        a.compile_options = options.as_ptr().cast();
        a.compile_options_size = options.len();
        self.0
            .plugin
            .check(unsafe { function!(self.0.plugin.api(), PJRT_Client_Compile)(&mut a) })?;
        self.loaded_executable(a.executable)
    }

    pub fn compile_hlo(&self, module: &[u8], options: &[u8]) -> Result<Executable> {
        self.compile(Program::hlo(module), options)
    }

    /// Load trusted native executable bytes from this platform and plugin version.
    ///
    /// # Safety
    /// The artifact must be trusted, unmodified, and compatible with this exact
    /// plugin/library version and execution platform. It may contain native code.
    /// This API does not verify provenance, signatures or hardware compatibility.
    pub unsafe fn deserialize_executable(&self, bytes: &[u8]) -> Result<Executable> {
        let mut a = args!(
            PJRT_Executable_DeserializeAndLoad_Args,
            PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE
        );
        a.client = self.0.raw;
        a.serialized_executable = bytes.as_ptr().cast();
        a.serialized_executable_size = bytes.len();
        self.0.plugin.check(unsafe {
            function!(self.0.plugin.api(), PJRT_Executable_DeserializeAndLoad)(&mut a)
        })?;
        self.loaded_executable(a.loaded_executable)
    }

    fn loaded_executable(&self, raw: *mut PJRT_LoadedExecutable) -> Result<Executable> {
        ensure!(
            !raw.is_null(),
            InvalidPluginDataSnafu {
                message: "null loaded executable",
            }
        );
        let mut exe = Executable {
            inner: Arc::new(ExecutableInner {
                client: self.0.clone(),
                raw,
                devices: Vec::new(),
            }),
            outputs: 0,
        };
        Arc::get_mut(&mut exe.inner)
            .expect("new executable is uniquely owned")
            .devices = executable_devices(&self.0, raw)?;
        let view = exe.unloaded()?;
        let mut count = args!(
            PJRT_Executable_NumOutputs_Args,
            PJRT_Executable_NumOutputs_Args_STRUCT_SIZE
        );
        count.executable = view.raw;
        self.0.plugin.check(unsafe {
            function!(self.0.plugin.api(), PJRT_Executable_NumOutputs)(&mut count)
        })?;
        exe.outputs = count.num_outputs;
        Ok(exe)
    }
}

// Validate compiled AND deserialized artifacts before exposing an executable.
fn executable_devices(
    client: &ClientInner,
    executable: *mut PJRT_LoadedExecutable,
) -> Result<Vec<*mut PJRT_Device>> {
    let plugin = &client.plugin;
    let mut a = args!(
        PJRT_LoadedExecutable_AddressableDevices_Args,
        PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE
    );
    a.executable = executable;
    plugin.check(unsafe {
        function!(plugin.api(), PJRT_LoadedExecutable_AddressableDevices)(&mut a)
    })?;
    ensure!(
        a.num_addressable_devices != 0,
        InvalidPluginDataSnafu {
            message: "executable has no addressable devices",
        }
    );
    ensure!(
        !a.addressable_devices.is_null(),
        InvalidPluginDataSnafu {
            message: "null executable device list",
        }
    );
    let devices =
        unsafe { std::slice::from_raw_parts(a.addressable_devices, a.num_addressable_devices) }
            .to_vec();
    ensure!(
        !devices.iter().any(|device| {
            device.is_null()
                || !client
                    .addressable_devices
                    .iter()
                    .any(|known| known == device)
        }),
        InvalidPluginDataSnafu {
            message: "executable references a non-addressable device",
        }
    );
    Ok(devices)
}

/// Raw-preserving PJRT buffer element type. Known constants are the types for
/// which rxla currently provides host transfer and HLO lowering support.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DType(PJRT_Buffer_Type);

impl DType {
    pub const U8: Self = Self(PJRT_Buffer_Type_PJRT_Buffer_Type_U8);
    pub const F16: Self = Self(PJRT_Buffer_Type_PJRT_Buffer_Type_F16);
    pub const F32: Self = Self(PJRT_Buffer_Type_PJRT_Buffer_Type_F32);
    pub const I32: Self = Self(PJRT_Buffer_Type_PJRT_Buffer_Type_S32);
    pub const BF16: Self = Self(PJRT_Buffer_Type_PJRT_Buffer_Type_BF16);

    pub const fn from_raw(raw: PJRT_Buffer_Type) -> Self {
        Self(raw)
    }

    pub const fn as_raw(self) -> PJRT_Buffer_Type {
        self.0
    }

    /// Storage width for the supported byte-addressable scalar types.
    pub const fn size_bytes(self) -> Option<usize> {
        match self {
            Self::U8 => Some(1),
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            _ => None,
        }
    }
}

impl std::fmt::Debug for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::U8 => f.write_str("U8"),
            Self::F16 => f.write_str("F16"),
            Self::F32 => f.write_str("F32"),
            Self::I32 => f.write_str("I32"),
            Self::BF16 => f.write_str("BF16"),
            Self(raw) => f.debug_tuple("DType").field(&raw).finish(),
        }
    }
}
mod sealed {
    pub trait Element {}
    impl Element for u8 {}
    impl Element for f32 {}
    impl Element for i32 {}
    impl Element for half::f16 {}
    impl Element for half::bf16 {}
}

/// A Rust scalar with an exact, byte-compatible PJRT host representation.
///
/// This trait is sealed so arbitrary types with padding or invalid bit patterns
/// cannot be used as native transfer destinations.
pub trait Element: sealed::Element + Copy + Default {
    const DTYPE: DType;
}
impl Element for u8 {
    const DTYPE: DType = DType::U8;
}
impl Element for f32 {
    const DTYPE: DType = DType::F32;
}
impl Element for i32 {
    const DTYPE: DType = DType::I32;
}
impl Element for half::f16 {
    const DTYPE: DType = DType::F16;
}
impl Element for half::bf16 {
    const DTYPE: DType = DType::BF16;
}

/// A shared device allocation handle.
///
/// Cloning is constant-time and keeps the native PJRT buffer alive; it does
/// not copy device memory.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

/// A backend-owned PJRT memory space kept alive by its client.
///
/// Memory kinds are backend-defined; common CUDA plugins expose device and
/// pinned-host spaces. Cloning this handle does not allocate or copy memory.
#[derive(Clone)]
pub struct Memory {
    client: Arc<ClientInner>,
    raw: NonNull<PJRT_Memory>,
}

// PJRT memory handles are immutable client-owned descriptors.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

/// A destination for typed host tensor uploads.
///
/// Implemented by [`Client`] for its selected device and by [`Memory`] for an
/// explicit backend memory space. Generic loaders can therefore support
/// accelerator-resident and offloaded checkpoints without duplicating their
/// decoding logic.
pub trait BufferDestination {
    fn buffer<T: Element>(&self, dims: &[i64], data: &[T]) -> Result<Buffer>;
}

impl BufferDestination for Client {
    fn buffer<T: Element>(&self, dims: &[i64], data: &[T]) -> Result<Buffer> {
        Client::buffer(self, dims, data)
    }
}

impl BufferDestination for Memory {
    fn buffer<T: Element>(&self, dims: &[i64], data: &[T]) -> Result<Buffer> {
        Client(self.client.clone()).buffer_in_memory(self, dims, data)
    }
}

/// Pins a PJRT buffer allocation for external pointer access.
///
/// The native external-reference count prevents the backend from deleting or
/// moving the allocation until this guard is dropped.
pub struct BufferExternalReference {
    buffer: Buffer,
    pointer: NonNull<std::ffi::c_void>,
    decrease: unsafe extern "C" fn(
        *mut PJRT_Buffer_DecreaseExternalReferenceCount_Args,
    ) -> *mut PJRT_Error,
}

// PJRT buffers and their external-reference accounting are thread-safe. The
// pointer is opaque and this type exposes no safe dereference operation.
unsafe impl Send for BufferExternalReference {}
unsafe impl Sync for BufferExternalReference {}

impl BufferExternalReference {
    /// Return the pinned backend pointer. Dereferencing it remains
    /// backend-specific and unsafe.
    pub fn as_ptr(&self) -> NonNull<std::ffi::c_void> {
        self.pointer
    }
}

impl Drop for BufferExternalReference {
    fn drop(&mut self) {
        let plugin = &self.buffer.inner.client.plugin;
        let mut args = args!(
            PJRT_Buffer_DecreaseExternalReferenceCount_Args,
            PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE
        );
        args.buffer = self.buffer.inner.raw.as_ptr();
        let _ = plugin.check(unsafe { (self.decrease)(&mut args) });
    }
}

struct BufferInner {
    client: Arc<ClientInner>,
    raw: NonNull<PJRT_Buffer>,
}

// PjRtBuffer is explicitly thread-safe in the OpenXLA interface. Destruction
// happens only after the last Arc, so it cannot race a safe Rust borrow.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}
impl Buffer {
    /// Whether PJRT reports that this allocation is host-accessible CPU memory.
    pub fn is_on_cpu(&self) -> Result<bool> {
        let plugin = &self.inner.client.plugin;
        let mut args = args!(
            PJRT_Buffer_IsOnCpu_Args,
            PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE
        );
        args.buffer = self.inner.raw.as_ptr();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_IsOnCpu)(&mut args) })?;
        Ok(args.is_on_cpu)
    }

    /// Pin the allocation and obtain its backend-specific opaque pointer.
    ///
    /// This may block in the backend. The returned guard must remain alive
    /// while the pointer is shared externally. It does not imply readiness,
    /// host accessibility, alignment, or any particular physical layout.
    pub fn external_reference(&self) -> Result<BufferExternalReference> {
        let plugin = &self.inner.client.plugin;
        let decrease_fn = function!(plugin.api(), PJRT_Buffer_DecreaseExternalReferenceCount);
        let mut increase = args!(
            PJRT_Buffer_IncreaseExternalReferenceCount_Args,
            PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE
        );
        increase.buffer = self.inner.raw.as_ptr();
        plugin.check(unsafe {
            function!(plugin.api(), PJRT_Buffer_IncreaseExternalReferenceCount)(&mut increase)
        })?;
        let mut args = args!(
            PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
            PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE
        );
        args.buffer = self.inner.raw.as_ptr();
        if let Err(error) = self.inner.client.plugin.check(unsafe {
            function!(
                self.inner.client.plugin.api(),
                PJRT_Buffer_OpaqueDeviceMemoryDataPointer
            )(&mut args)
        }) {
            let mut decrease_args = args!(
                PJRT_Buffer_DecreaseExternalReferenceCount_Args,
                PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE
            );
            decrease_args.buffer = self.inner.raw.as_ptr();
            let _ = plugin.check(unsafe { decrease_fn(&mut decrease_args) });
            return Err(error);
        }
        let pointer = match NonNull::new(args.device_memory_ptr) {
            Some(pointer) => pointer,
            None => {
                let mut decrease_args = args!(
                    PJRT_Buffer_DecreaseExternalReferenceCount_Args,
                    PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE
                );
                decrease_args.buffer = self.inner.raw.as_ptr();
                let _ = plugin.check(unsafe { decrease_fn(&mut decrease_args) });
                return InvalidPluginDataSnafu {
                    message: "PJRT returned a null opaque device pointer",
                }
                .fail();
            }
        };
        Ok(BufferExternalReference {
            buffer: self.clone(),
            pointer,
            decrease: decrease_fn,
        })
    }

    /// Whether two handles retain the same native PJRT allocation.
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn belongs_to(&self, client: &Client) -> bool {
        Arc::ptr_eq(&self.inner.client, &client.0)
    }

    /// Wait until all work producing this buffer has completed.
    ///
    /// Copies and executable outputs may return a handle before their device
    /// work finishes. A serving scheduler can enqueue a prefetch, perform
    /// independent CPU work, and then account for only the unhidden transfer
    /// tail at this synchronization point.
    pub fn block_until_ready(&self) -> Result<()> {
        let plugin = &self.inner.client.plugin;
        let event = pjrt_call!(
            plugin,
            PJRT_Buffer_ReadyEvent,
            PJRT_Buffer_ReadyEvent_Args,
            PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE,
            buffer = self.inner.raw.as_ptr(),
        );
        plugin.wait(event.event)
    }

    /// Copy this buffer to another addressable device owned by the same client.
    ///
    /// PJRT performs the transfer directly; no host allocation or download is
    /// involved. Copying to the buffer's current device returns a shared handle
    /// to the existing allocation.
    pub fn copy_to_device(&self, device_index: usize) -> Result<Buffer> {
        let device = *self
            .inner
            .client
            .addressable_devices
            .get(device_index)
            .context(InvalidArgumentSnafu {
                message: "addressable device index out of range",
            })?;
        if self.device_index()? == device_index {
            return Ok(self.clone());
        }
        let plugin = &self.inner.client.plugin;
        let mut args = args!(
            PJRT_Buffer_CopyToDevice_Args,
            PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE
        );
        args.buffer = self.inner.raw.as_ptr();
        args.dst_device = device;
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_CopyToDevice)(&mut args) })?;
        Ok(Buffer {
            inner: Arc::new(BufferInner::new(
                self.inner.client.clone(),
                args.dst_buffer,
            )?),
        })
    }

    /// Asynchronously copy this buffer into another memory space of its client.
    ///
    /// Readiness remains represented by the returned PJRT buffer. This is the
    /// primitive used by residency managers to evict device buffers to pinned
    /// host memory and prefetch them back without a host round trip through
    /// Rust-owned bytes.
    pub fn copy_to_memory(&self, memory: &Memory) -> Result<Buffer> {
        ensure!(
            Arc::ptr_eq(&self.inner.client, &memory.client),
            InvalidArgumentSnafu {
                message: "buffer and destination memory belong to different clients",
            }
        );
        let current = pjrt_call!(
            &self.inner.client.plugin,
            PJRT_Buffer_Memory,
            PJRT_Buffer_Memory_Args,
            PJRT_Buffer_Memory_Args_STRUCT_SIZE,
            buffer = self.inner.raw.as_ptr(),
        );
        if current.memory == memory.raw.as_ptr() {
            return Ok(self.clone());
        }
        let plugin = &self.inner.client.plugin;
        let mut args = args!(
            PJRT_Buffer_CopyToMemory_Args,
            PJRT_Buffer_CopyToMemory_Args_STRUCT_SIZE
        );
        args.buffer = self.inner.raw.as_ptr();
        args.dst_memory = memory.raw.as_ptr();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_CopyToMemory)(&mut args) })?;
        Ok(Buffer {
            inner: Arc::new(BufferInner::new(
                self.inner.client.clone(),
                args.dst_buffer,
            )?),
        })
    }

    /// Prefer a native same-client device copy and otherwise stage through host.
    pub fn copy_to_device_or_via_host(
        &self,
        destination: &Client,
        device_index: usize,
    ) -> Result<Buffer> {
        if self.belongs_to(destination) {
            self.copy_to_device(device_index)
        } else {
            self.copy_to_device_via_host(destination, device_index)
        }
    }

    /// Like [`Self::copy_to_device_or_via_host`], with a host-staging limit.
    /// Native same-client copies do not consume host payload budget.
    pub fn copy_to_device_or_via_host_with_limit(
        &self,
        destination: &Client,
        device_index: usize,
        max_host_bytes: usize,
    ) -> Result<Buffer> {
        if self.belongs_to(destination) {
            self.copy_to_device(device_index)
        } else {
            self.copy_to_device_via_host_with_limit(destination, device_index, max_host_bytes)
        }
    }

    /// Copy into the destination client's selected device through host memory.
    /// Preserves shape and F32/I32/BF16 dtype (BF16 is copied as raw bits).
    /// Downloads synchronously and uploads a new buffer, even on the same client.
    /// This is not a device-to-device, zero-copy or asynchronous transfer; host
    /// temporary storage is one full tensor. Neither source ownership nor state
    /// changes on failure. Persistent native handles are safe to share across
    /// threads; the synchronous host temporary remains local to this call.
    pub fn copy_to_client_via_host(&self, destination: &Client) -> Result<Buffer> {
        self.copy_to_client_via_host_with_limit(destination, usize::MAX)
    }
    /// Copy through host memory into one addressable device of the destination
    /// client. The index uses the same ordering as [`Client::info`].
    pub fn copy_to_device_via_host(
        &self,
        destination: &Client,
        device_index: usize,
    ) -> Result<Buffer> {
        self.copy_to_device_via_host_with_limit(destination, device_index, usize::MAX)
    }
    /// Like [`Self::copy_to_client_via_host`], with a per-copy host payload limit.
    /// Queries the native required download size and rejects excess before
    /// allocating/downloading payload or uploading the destination. Metadata
    /// queries may still call the plugin. Zero permits only empty payloads.
    /// This does not bound plugin-internal memory, allocator overhead, device
    /// memory or the combined memory of concurrent copies.
    pub fn copy_to_client_via_host_with_limit(
        &self,
        destination: &Client,
        max_host_bytes: usize,
    ) -> Result<Buffer> {
        let selected = destination
            .0
            .addressable_devices
            .iter()
            .position(|&device| device == destination.0.device)
            .expect("selected device belongs to addressable devices");
        self.copy_to_device_via_host_with_limit(destination, selected, max_host_bytes)
    }

    pub fn copy_to_device_via_host_with_limit(
        &self,
        destination: &Client,
        device_index: usize,
        max_host_bytes: usize,
    ) -> Result<Buffer> {
        let dtype = self.dtype()?;
        let dimensions = self.dimensions()?;
        match dtype {
            DType::U8 => destination.buffer_on_device(
                device_index,
                &dimensions,
                &self.to_vec_with_limit::<u8>(max_host_bytes)?,
            ),
            DType::F16 => destination.buffer_on_device(
                device_index,
                &dimensions,
                &self.to_vec_with_limit::<half::f16>(max_host_bytes)?,
            ),
            DType::F32 => destination.buffer_on_device(
                device_index,
                &dimensions,
                &self.to_vec_with_limit::<f32>(max_host_bytes)?,
            ),
            DType::I32 => destination.buffer_on_device(
                device_index,
                &dimensions,
                &self.to_vec_with_limit::<i32>(max_host_bytes)?,
            ),
            DType::BF16 => destination.buffer_on_device(
                device_index,
                &dimensions,
                &self.to_vec_with_limit::<half::bf16>(max_host_bytes)?,
            ),
            dtype => UnsupportedSnafu {
                message: format!("host transfer for PJRT buffer type {}", dtype.as_raw()),
            }
            .fail(),
        }
    }
    pub fn device_index(&self) -> Result<usize> {
        let plugin = &self.inner.client.plugin;
        let mut a = args!(PJRT_Buffer_Device_Args, PJRT_Buffer_Device_Args_STRUCT_SIZE);
        a.buffer = self.inner.raw.as_ptr();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_Device)(&mut a) })?;
        self.inner
            .client
            .addressable_devices
            .iter()
            .position(|&device| device == a.device)
            .context(InvalidPluginDataSnafu {
                message: "buffer belongs to a non-addressable device",
            })
    }
    /// Query the native host-download payload size without allocating or copying
    /// the payload. This is not device allocation size or plugin memory usage.
    pub fn host_payload_bytes(&self) -> Result<usize> {
        Ok(match self.dtype()? {
            DType::U8 => self.download_args::<u8>()?.dst_size,
            DType::F16 => self.download_args::<half::f16>()?.dst_size,
            DType::F32 => self.download_args::<f32>()?.dst_size,
            DType::I32 => self.download_args::<i32>()?.dst_size,
            DType::BF16 => self.download_args::<half::bf16>()?.dst_size,
            dtype => {
                return UnsupportedSnafu {
                    message: format!("host transfer for PJRT buffer type {}", dtype.as_raw()),
                }
                .fail();
            }
        })
    }
    pub fn dimensions(&self) -> Result<Vec<i64>> {
        let plugin = &self.inner.client.plugin;
        let mut a = args!(
            PJRT_Buffer_Dimensions_Args,
            PJRT_Buffer_Dimensions_Args_STRUCT_SIZE
        );
        a.buffer = self.inner.raw.as_ptr();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_Dimensions)(&mut a) })?;
        if a.num_dims == 0 {
            return Ok(Vec::new());
        }
        ensure!(
            !a.dims.is_null(),
            InvalidPluginDataSnafu {
                message: "null buffer dimensions",
            }
        );
        Ok(unsafe { std::slice::from_raw_parts(a.dims, a.num_dims) }.to_vec())
    }
    pub fn dtype(&self) -> Result<DType> {
        let plugin = &self.inner.client.plugin;
        let mut ty = args!(
            PJRT_Buffer_ElementType_Args,
            PJRT_Buffer_ElementType_Args_STRUCT_SIZE
        );
        ty.buffer = self.inner.raw.as_ptr();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_ElementType)(&mut ty) })?;
        Ok(DType::from_raw(ty.type_))
    }
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        self.to_vec_with_limit(usize::MAX)
    }
    fn to_vec_with_limit<T: Element>(&self, max_host_bytes: usize) -> Result<Vec<T>> {
        let _span = tracing::trace_span!("pjrt.device_to_host", dtype = ?T::DTYPE).entered();
        let a = self.download_args::<T>()?;
        if a.dst_size > max_host_bytes {
            return LimitExceededSnafu {
                message: format!(
                    "host transfer needs {} bytes, exceeds limit {max_host_bytes}",
                    a.dst_size
                ),
            }
            .fail();
        }
        let elements = a.dst_size / std::mem::size_of::<T>();
        let mut data = Vec::new();
        data.try_reserve_exact(elements)
            .context(ResourceExhaustedSnafu {
                operation: "host download allocation",
            })?;
        data.resize(elements, T::default());
        self.download(a, &mut data)?;
        Ok(data)
    }
    pub fn copy_to<T: Element>(&self, dst: &mut [T]) -> Result<()> {
        let _span = tracing::trace_span!("pjrt.device_to_host", dtype = ?T::DTYPE).entered();
        self.download(self.download_args::<T>()?, dst)
    }
    fn download_args<T: Element>(&self) -> Result<PJRT_Buffer_ToHostBuffer_Args> {
        let plugin = &self.inner.client.plugin;
        ensure!(
            self.dtype()? == T::DTYPE,
            InvalidArgumentSnafu {
                message: format!("buffer is not {:?}", T::DTYPE),
            }
        );
        let mut a = args!(
            PJRT_Buffer_ToHostBuffer_Args,
            PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE
        );
        a.src = self.inner.raw.as_ptr();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_ToHostBuffer)(&mut a) })?;
        ensure!(
            a.dst_size % std::mem::size_of::<T>() == 0,
            InvalidPluginDataSnafu {
                message: "unaligned element byte count",
            }
        );
        Ok(a)
    }
    fn download<T: Element>(
        &self,
        mut a: PJRT_Buffer_ToHostBuffer_Args,
        dst: &mut [T],
    ) -> Result<()> {
        if std::mem::size_of_val(dst) != a.dst_size {
            return InvalidArgumentSnafu {
                message: format!(
                    "destination has {} elements, expected {}",
                    dst.len(),
                    a.dst_size / std::mem::size_of::<T>()
                ),
            }
            .fail();
        }
        let plugin = &self.inner.client.plugin;
        a.dst = dst.as_mut_ptr().cast();
        plugin.check(unsafe { function!(plugin.api(), PJRT_Buffer_ToHostBuffer)(&mut a) })?;
        plugin.wait(a.event)?;
        Ok(())
    }
}
impl BufferInner {
    fn new(client: Arc<ClientInner>, raw: *mut PJRT_Buffer) -> Result<Self> {
        let raw = NonNull::new(raw).context(InvalidPluginDataSnafu {
            message: "PJRT returned a null buffer",
        })?;
        Ok(Self { client, raw })
    }

    fn destroy(&self) -> Result<()> {
        let mut a = args!(
            PJRT_Buffer_Destroy_Args,
            PJRT_Buffer_Destroy_Args_STRUCT_SIZE
        );
        a.buffer = self.raw.as_ptr();
        self.client
            .plugin
            .check(unsafe { function!(self.client.plugin.api(), PJRT_Buffer_Destroy)(&mut a) })
    }
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

pub struct Executable {
    inner: Arc<ExecutableInner>,
    outputs: usize,
}
struct ExecutableInner {
    client: Arc<ClientInner>,
    raw: *mut PJRT_LoadedExecutable,
    devices: Vec<*mut PJRT_Device>,
}

// PJRT permits concurrent execution of one loaded executable. Device pointers
// are immutable client-owned metadata retained by `client`.
unsafe impl Send for ExecutableInner {}
unsafe impl Sync for ExecutableInner {}

/// Owned optimized compiler IR, not a serialized native executable artifact.
pub struct OptimizedProgram {
    pub format: String,
    pub code: Vec<u8>,
}
// A separate PJRT_Executable reference must be destroyed independently of the
// loaded executable. Keep both the plugin and its destructor alive during use.
struct ExecutableView {
    raw: *mut PJRT_Executable,
    plugin: Plugin,
    destroy: unsafe extern "C" fn(*mut PJRT_Executable_Destroy_Args) -> *mut PJRT_Error,
}
impl Drop for ExecutableView {
    fn drop(&mut self) {
        let mut a = args!(
            PJRT_Executable_Destroy_Args,
            PJRT_Executable_Destroy_Args_STRUCT_SIZE
        );
        a.executable = self.raw;
        let _ = self.plugin.check(unsafe { (self.destroy)(&mut a) });
    }
}

impl Executable {
    pub fn device_count(&self) -> usize {
        self.inner.devices.len()
    }

    /// Request an optimized program in a plugin-supported format, such as `hlo`.
    /// The plugin may return a different format; inspect the returned label.
    /// No compilation/execution is requested. This optional API can be unsupported.
    pub fn optimized_program(&self, format: &str) -> Result<OptimizedProgram> {
        let view = self.unloaded()?;
        let plugin = &self.inner.client.plugin;
        let get = function!(plugin.api(), PJRT_Executable_OptimizedProgram);
        let mut program = args!(PJRT_Program, PJRT_Program_STRUCT_SIZE);
        program.format = format.as_ptr().cast();
        program.format_size = format.len();
        let mut a = args!(
            PJRT_Executable_OptimizedProgram_Args,
            PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE
        );
        a.executable = view.raw;
        a.program = &mut program;
        plugin.check(unsafe { get(&mut a) })?;
        ensure!(
            program.code_size != 0,
            InvalidPluginDataSnafu {
                message: "empty optimized program",
            }
        );
        let mut code = Vec::new();
        code.try_reserve_exact(program.code_size)
            .context(ResourceExhaustedSnafu {
                operation: "optimized program allocation",
            })?;
        code.resize(program.code_size, 0u8);
        program.code = code.as_mut_ptr().cast();
        plugin.check(unsafe { get(&mut a) })?;
        ensure!(
            program.code_size <= code.len(),
            InvalidPluginDataSnafu {
                message: "optimized program size changed during retrieval",
            }
        );
        code.truncate(program.code_size);
        ensure!(
            !program.format.is_null() && program.format_size != 0,
            InvalidPluginDataSnafu {
                message: "missing optimized program format",
            }
        );
        let format = std::str::from_utf8(unsafe {
            std::slice::from_raw_parts(program.format.cast(), program.format_size)
        })
        .context(InvalidPluginEncodingSnafu {
            operation: "optimized program format",
        })?
        .to_owned();
        Ok(OptimizedProgram { format, code })
    }

    fn unloaded(&self) -> Result<ExecutableView> {
        let plugin = &self.inner.client.plugin;
        let destroy = function!(plugin.api(), PJRT_Executable_Destroy);
        let mut a = args!(
            PJRT_LoadedExecutable_GetExecutable_Args,
            PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE
        );
        a.loaded_executable = self.inner.raw;
        plugin.check(unsafe {
            function!(plugin.api(), PJRT_LoadedExecutable_GetExecutable)(&mut a)
        })?;
        ensure!(
            !a.executable.is_null(),
            InvalidPluginDataSnafu {
                message: "null executable reference",
            }
        );
        Ok(ExecutableView {
            raw: a.executable,
            plugin: plugin.clone(),
            destroy,
        })
    }

    /// Platform-specific native executable artifact, not a portable HLO module.
    /// Store only in a trusted location and pin the producing plugin/platform.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let view = self.unloaded()?;
        let plugin = &self.inner.client.plugin;
        let mut a = args!(
            PJRT_Executable_Serialize_Args,
            PJRT_Executable_Serialize_Args_STRUCT_SIZE
        );
        a.executable = view.raw;
        plugin.check(unsafe { function!(plugin.api(), PJRT_Executable_Serialize)(&mut a) })?;
        let deleter = a
            .serialized_executable_deleter
            .context(InvalidPluginDataSnafu {
                message: "null serialized executable deleter",
            })?;
        struct Serialized {
            raw: *mut PJRT_SerializedExecutable,
            deleter: unsafe extern "C" fn(*mut PJRT_SerializedExecutable),
        }
        impl Drop for Serialized {
            fn drop(&mut self) {
                unsafe { (self.deleter)(self.raw) };
            }
        }
        ensure!(
            !a.serialized_executable.is_null(),
            InvalidPluginDataSnafu {
                message: "null serialized executable owner",
            }
        );
        let _owner = Serialized {
            raw: a.serialized_executable,
            deleter,
        };
        ensure!(
            a.serialized_bytes_size != 0,
            InvalidPluginDataSnafu {
                message: "empty serialized executable",
            }
        );
        ensure!(
            !a.serialized_bytes.is_null(),
            InvalidPluginDataSnafu {
                message: "null serialized executable bytes",
            }
        );
        Ok(unsafe {
            std::slice::from_raw_parts(a.serialized_bytes.cast(), a.serialized_bytes_size)
        }
        .to_vec())
    }

    pub fn execute(&self, inputs: &[&Buffer]) -> Result<Vec<Buffer>> {
        let _span = tracing::debug_span!(
            "pjrt.execute",
            inputs = inputs.len(),
            outputs = self.outputs
        )
        .entered();
        self.submit(inputs)?.wait()
    }

    /// Submit without an explicit device-completion wait. The plugin may still
    /// block during submission; neither asynchronous kernels nor overlap is
    /// guaranteed. All inputs are non-donatable and their native owners are
    /// retained until completion. Original Buffer/Executable/Client handles may
    /// be dropped after submission. PendingExecution is thread-affine, not Future.
    pub fn submit(&self, inputs: &[&Buffer]) -> Result<PendingExecution> {
        let _span =
            tracing::debug_span!("pjrt.submit", inputs = inputs.len(), outputs = self.outputs)
                .entered();
        ensure!(
            self.inner.devices.len() == 1,
            InvalidStateSnafu {
                message: "multi-device executable requires execute_sharded",
            }
        );
        ensure!(
            !inputs
                .iter()
                .any(|b| !Arc::ptr_eq(&b.inner.client, &self.inner.client)),
            InvalidArgumentSnafu {
                message: "buffers belong to a different client",
            }
        );
        let retained_inputs: Vec<_> = inputs.iter().map(|b| b.inner.clone()).collect();
        let plugin = &self.inner.client.plugin;
        let pointers: Vec<_> = inputs.iter().map(|b| b.inner.raw.as_ptr()).collect();
        let input_list = pointers.as_ptr();
        let mut output = vec![ptr::null_mut(); self.outputs];
        let output_list = output.as_mut_ptr();
        let non_donated: Vec<i64> = (0..inputs.len() as i64).collect();
        let mut options = args!(PJRT_ExecuteOptions, PJRT_ExecuteOptions_STRUCT_SIZE);
        options.non_donatable_input_indices = non_donated.as_ptr();
        options.num_non_donatable_input_indices = non_donated.len();
        let mut event = ptr::null_mut();
        let mut a = args!(
            PJRT_LoadedExecutable_Execute_Args,
            PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
        );
        a.executable = self.inner.raw;
        a.options = &mut options;
        a.argument_lists = &input_list;
        a.num_devices = 1;
        a.num_args = inputs.len();
        a.output_lists = &output_list;
        a.device_complete_events = &mut event;
        plugin.check(unsafe { function!(plugin.api(), PJRT_LoadedExecutable_Execute)(&mut a) })?;
        let buffers: Vec<_> = output
            .into_iter()
            .map(|raw| {
                Ok(Buffer {
                    inner: Arc::new(BufferInner::new(self.inner.client.clone(), raw)?),
                })
            })
            .collect::<Result<_>>()?;
        Ok(PendingExecution {
            executable: self.inner.clone(),
            _inputs: retained_inputs,
            outputs: buffers,
            event: Some(event),
        })
    }

    /// Execute one argument list per addressable device and return one output
    /// list per device. Device ordering is the executable's PJRT ordering.
    /// This synchronous boundary waits for every completion event, including
    /// after an earlier device reports an error.
    pub fn execute_sharded(&self, inputs: &[&[&Buffer]]) -> Result<Vec<Vec<Buffer>>> {
        let device_count = self.inner.devices.len();
        ensure!(
            inputs.len() == device_count,
            InvalidArgumentSnafu {
                message: "sharded argument-list count mismatch",
            }
        );
        let num_args = inputs.first().map_or(0, |values| values.len());
        ensure!(
            !inputs.iter().any(|values| values.len() != num_args),
            InvalidArgumentSnafu {
                message: "sharded argument lists have different arity",
            }
        );
        ensure!(
            !inputs
                .iter()
                .flat_map(|values| values.iter())
                .any(|buffer| !Arc::ptr_eq(&buffer.inner.client, &self.inner.client)),
            InvalidArgumentSnafu {
                message: "buffers belong to a different client",
            }
        );

        let argument_storage = inputs
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|buffer| buffer.inner.raw.as_ptr())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let argument_lists = argument_storage
            .iter()
            .map(|values| values.as_ptr())
            .collect::<Vec<_>>();
        let mut output_storage = (0..device_count)
            .map(|_| vec![ptr::null_mut(); self.outputs])
            .collect::<Vec<_>>();
        let output_lists = output_storage
            .iter_mut()
            .map(|values| values.as_mut_ptr())
            .collect::<Vec<_>>();
        let non_donated = (0..num_args as i64).collect::<Vec<_>>();
        let mut options = args!(PJRT_ExecuteOptions, PJRT_ExecuteOptions_STRUCT_SIZE);
        options.non_donatable_input_indices = non_donated.as_ptr();
        options.num_non_donatable_input_indices = non_donated.len();
        let mut events = vec![ptr::null_mut(); device_count];
        let mut a = args!(
            PJRT_LoadedExecutable_Execute_Args,
            PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
        );
        a.executable = self.inner.raw;
        a.options = &mut options;
        a.argument_lists = argument_lists.as_ptr();
        a.num_devices = device_count;
        a.num_args = num_args;
        a.output_lists = output_lists.as_ptr();
        a.device_complete_events = events.as_mut_ptr();
        let plugin = &self.inner.client.plugin;
        plugin.check(unsafe { function!(plugin.api(), PJRT_LoadedExecutable_Execute)(&mut a) })?;

        let outputs = output_storage
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .map(|raw| {
                        Ok(Buffer {
                            inner: Arc::new(BufferInner::new(self.inner.client.clone(), raw)?),
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut failure = None;
        for event in events {
            if let Err(err) = plugin.wait(event)
                && failure.is_none()
            {
                failure = Some(err);
            }
        }
        if let Some(err) = failure {
            return Err(err);
        }
        Ok(outputs)
    }
}
impl ExecutableInner {
    fn destroy(&self) -> Result<()> {
        let mut a = args!(
            PJRT_LoadedExecutable_Destroy_Args,
            PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE
        );
        a.executable = self.raw;
        self.client.plugin.check(unsafe {
            function!(self.client.plugin.api(), PJRT_LoadedExecutable_Destroy)(&mut a)
        })
    }
}
impl Drop for ExecutableInner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

fn dma_unmap(client: &ClientInner, data: *mut std::ffi::c_void) -> Result<()> {
    let mut unmap = args!(
        PJRT_Client_DmaUnmap_Args,
        PJRT_Client_DmaUnmap_Args_STRUCT_SIZE
    );
    unmap.client = client.raw;
    unmap.data = data;
    client
        .plugin
        .check(unsafe { function!(client.plugin.api(), PJRT_Client_DmaUnmap)(&mut unmap) })
}

/// An in-flight upload from an owned DMA-mapped host allocation.
///
/// The buffer handle can be submitted immediately: PJRT carries the readiness
/// dependency into downstream execution. `wait` observes transfer errors and
/// releases the host mapping. Dropping also waits, then releases it.
#[must_use = "retain the pinned allocation until upload completion"]
pub struct PendingHostUpload<T: Element> {
    client: Arc<ClientInner>,
    data: Option<Vec<T>>,
    buffer: Option<Buffer>,
    event: Option<*mut PJRT_Event>,
    mapped: bool,
}

impl<T: Element> PendingHostUpload<T> {
    pub fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().expect("buffer exists until wait")
    }

    fn finish(&mut self) -> Result<()> {
        let transfer = match self.event.take() {
            Some(event) => self.client.plugin.wait(event),
            None => Ok(()),
        };
        let unmap = if self.mapped {
            self.mapped = false;
            let data = self.data.as_mut().expect("host allocation retained");
            dma_unmap(&self.client, data.as_mut_ptr().cast())
        } else {
            Ok(())
        };
        transfer.and(unmap)
    }

    pub fn wait(mut self) -> Result<Buffer> {
        self.finish()?;
        self.data.take();
        Ok(self.buffer.take().expect("buffer exists until wait"))
    }
}

impl<T: Element> Drop for PendingHostUpload<T> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Owned in-flight execution. `wait` reports completion errors and yields the
/// outputs; Drop waits and discards errors/results, so dropping can block.
/// This is not cancellation, a Future, or a Send/Sync handle. Forgetting this
/// object leaks all retained native owners rather than releasing in-flight data.
///
/// ```compile_fail
/// fn needs_send<T: Send>() {}
/// needs_send::<rxla_pjrt::PendingExecution>();
/// ```
/// ```compile_fail
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<rxla_pjrt::PendingExecution>();
/// ```
#[must_use = "wait for execution errors/results; dropping also waits"]
pub struct PendingExecution {
    executable: Arc<ExecutableInner>,
    _inputs: Vec<Arc<BufferInner>>,
    outputs: Vec<Buffer>,
    event: Option<*mut PJRT_Event>,
}
impl PendingExecution {
    /// Borrow output handles before completion, for submission as inputs to
    /// another executable on the same client. PJRT tracks buffer dependencies;
    /// this accessor does not wait, download data, or imply successful completion.
    /// A downstream submission retains these buffers independently of this borrow.
    /// Host downloads still wait for transfer completion. Keep and wait on each
    /// pending task to observe its execution status; dropping it still waits.
    pub fn outputs(&self) -> &[Buffer] {
        &self.outputs
    }

    /// Query completion without waiting or consuming this handle. `true` also
    /// includes failed executions: call `wait` to retrieve the completion status
    /// and outputs. Query errors leave ownership intact; `wait`/Drop still work.
    /// This optional plugin API does not register a wakeup. Avoid busy polling.
    pub fn is_ready(&self) -> Result<bool> {
        let event = self.event.context(InvalidStateSnafu {
            message: "execution completion already consumed",
        })?;
        ensure!(
            !event.is_null(),
            InvalidPluginDataSnafu {
                message: "plugin returned a null completion event",
            }
        );
        let plugin = &self.executable.client.plugin;
        let mut a = args!(PJRT_Event_IsReady_Args, PJRT_Event_IsReady_Args_STRUCT_SIZE);
        a.event = event;
        plugin.check(unsafe { function!(plugin.api(), PJRT_Event_IsReady)(&mut a) })?;
        Ok(a.is_ready)
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(event) = self.event.take() {
            self.executable.client.plugin.wait(event)?;
        }
        Ok(())
    }
    pub fn wait(mut self) -> Result<Vec<Buffer>> {
        let _span = tracing::debug_span!("pjrt.wait").entered();
        self.finish()?;
        Ok(std::mem::take(&mut self.outputs))
    }
}
impl Drop for PendingExecution {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod lifetime_tests {
    use super::*;

    #[test]
    fn dtype_preserves_unknown_pjrt_values() {
        let unknown = DType::from_raw(u32::MAX);
        assert_eq!(unknown.as_raw(), u32::MAX);
        assert_eq!(unknown.size_bytes(), None);
        assert_eq!(format!("{unknown:?}"), "DType(4294967295)");
        assert_eq!(DType::F32.size_bytes(), Some(4));
        assert_eq!(format!("{:?}", DType::BF16), "BF16");
    }

    #[test]
    fn native_half_types_have_exact_pjrt_layouts() {
        assert_eq!(DType::F16.size_bytes(), Some(size_of::<half::f16>()));
        assert_eq!(DType::BF16.size_bytes(), Some(size_of::<half::bf16>()));
        assert_eq!(half::f16::DTYPE, DType::F16);
        assert_eq!(half::bf16::DTYPE, DType::BF16);
    }

    #[test]
    fn pjrt_status_codes_remain_machine_readable() {
        let retryable = Error::Pjrt {
            code: PjrtErrorCode::Unavailable,
            message: "device temporarily unavailable".into(),
        };
        assert_eq!(retryable.pjrt_code(), Some(PjrtErrorCode::Unavailable));
        assert!(retryable.is_retryable());
        assert!(
            !Error::InvalidArgument {
                message: "bad shape".into(),
            }
            .is_retryable()
        );
        assert_eq!(
            PjrtErrorCode::from(PJRT_Error_Code_PJRT_Error_Code_DATA_LOSS),
            PjrtErrorCode::DataLoss
        );
        assert_eq!(
            PjrtErrorCode::from(u32::MAX),
            PjrtErrorCode::Unrecognized(u32::MAX)
        );
    }

    #[test]
    fn linked_api_registration_is_named_and_does_not_need_a_library() {
        unsafe extern "C" fn initialize(_: *mut PJRT_Plugin_Initialize_Args) -> *mut PJRT_Error {
            ptr::null_mut()
        }
        let mut api = lifetime_table();
        api.PJRT_Plugin_Initialize = Some(initialize);
        let api = Box::leak(Box::new(api));
        let mut plugins = PluginRegistry::new();
        unsafe { plugins.register_api("linked", api) }.unwrap();
        assert_eq!(plugins.names().collect::<Vec<_>>(), ["linked"]);
    }

    #[test]
    #[cfg(unix)]
    fn readiness_queries_do_not_consume_events_even_on_error() {
        use std::cell::Cell;
        #[derive(Default)]
        struct Event {
            ready: Cell<bool>,
            query_error: Cell<bool>,
            waits: Cell<usize>,
            destroys: Cell<usize>,
        }
        unsafe extern "C" fn ready(a: *mut PJRT_Event_IsReady_Args) -> *mut PJRT_Error {
            let a = unsafe { &mut *a };
            let event = unsafe { &*a.event.cast::<Event>() };
            if event.query_error.get() {
                // Error_Message/Destroy below are no-op stubs; neither
                // dereferences this synthetic error token.
                return a.event.cast();
            }
            a.is_ready = event.ready.get();
            ptr::null_mut()
        }
        unsafe extern "C" fn wait(a: *mut PJRT_Event_Await_Args) -> *mut PJRT_Error {
            let event = unsafe { &*(*a).event.cast::<Event>() };
            event.waits.set(event.waits.get() + 1);
            ptr::null_mut()
        }
        unsafe extern "C" fn destroy(a: *mut PJRT_Event_Destroy_Args) -> *mut PJRT_Error {
            let event = unsafe { &*(*a).event.cast::<Event>() };
            event.destroys.set(event.destroys.get() + 1);
            ptr::null_mut()
        }
        for missing in [false, true] {
            let event = Event::default();
            let mut api = lifetime_table();
            api.PJRT_Event_IsReady = if missing { None } else { Some(ready) };
            api.PJRT_Event_Await = Some(wait);
            api.PJRT_Event_Destroy = Some(destroy);
            // No native plugin is loaded. The table and synthetic event outlive
            // all owners; the other lifetime-table destructors are no-op stubs.
            let client = Arc::new(ClientInner {
                plugin: Plugin(Arc::new(PluginInner {
                    api: &api,
                    _library: Some(ManuallyDrop::new(
                        libloading::os::unix::Library::this().into(),
                    )),
                })),
                raw: ptr::null_mut(),
                device: ptr::null_mut(),
                addressable_devices: vec![],
                _key_value_callbacks: None,
            });
            let pending = PendingExecution {
                executable: Arc::new(ExecutableInner {
                    client,
                    raw: ptr::null_mut(),
                    devices: vec![],
                }),
                _inputs: vec![],
                outputs: vec![],
                event: Some(ptr::from_ref(&event).cast_mut().cast()),
            };
            if missing {
                assert!(matches!(
                    pending.is_ready().unwrap_err(),
                    Error::IncompatiblePlugin { message }
                        if message.contains("PJRT_Event_IsReady")
                ));
                drop(pending);
            } else {
                assert!(!pending.is_ready().unwrap());
                event.query_error.set(true);
                assert!(pending.is_ready().is_err());
                assert_eq!(event.waits.get(), 0);
                assert_eq!(event.destroys.get(), 0);
                event.query_error.set(false);
                event.ready.set(true);
                assert!(pending.is_ready().unwrap());
                assert!(pending.is_ready().unwrap());
                assert!(pending.wait().unwrap().is_empty());
            }
            assert_eq!(event.waits.get(), 1);
            assert_eq!(event.destroys.get(), 1);
        }
    }

    #[test]
    fn host_copy_budget_rejects_native_size_before_payload_or_upload() {
        use std::cell::Cell;
        struct Probe {
            dtype: DType,
            queries: Cell<usize>,
            downloads: Cell<usize>,
            uploads: Cell<usize>,
        }
        unsafe extern "C" fn dtype(a: *mut PJRT_Buffer_ElementType_Args) -> *mut PJRT_Error {
            let a = unsafe { &mut *a };
            let probe = unsafe { &*a.buffer.cast::<Probe>() };
            a.type_ = probe.dtype.as_raw();
            ptr::null_mut()
        }
        unsafe extern "C" fn dimensions(a: *mut PJRT_Buffer_Dimensions_Args) -> *mut PJRT_Error {
            // Deliberately scalar metadata: use native download size, not a
            // shape-derived assumption, to decide the host allocation budget.
            unsafe {
                (*a).num_dims = 0;
            }
            ptr::null_mut()
        }
        unsafe extern "C" fn download(a: *mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PJRT_Error {
            let a = unsafe { &mut *a };
            let probe = unsafe { &*a.src.cast::<Probe>() };
            if a.dst.is_null() {
                probe.queries.set(probe.queries.get() + 1);
                // Aligned for every supported dtype, impossible to allocate.
                a.dst_size = usize::MAX - 3;
            } else {
                probe.downloads.set(probe.downloads.get() + 1);
            }
            ptr::null_mut()
        }
        unsafe extern "C" fn upload(
            a: *mut PJRT_Client_BufferFromHostBuffer_Args,
        ) -> *mut PJRT_Error {
            let probe = unsafe { &*(*a).client.cast::<Probe>() };
            probe.uploads.set(probe.uploads.get() + 1);
            ptr::null_mut()
        }
        for kind in [DType::F32, DType::I32, DType::BF16] {
            let probe = Probe {
                dtype: kind,
                queries: Cell::new(0),
                downloads: Cell::new(0),
                uploads: Cell::new(0),
            };
            let mut api = lifetime_table();
            api.PJRT_Buffer_ElementType = Some(dtype);
            api.PJRT_Buffer_Dimensions = Some(dimensions);
            api.PJRT_Buffer_ToHostBuffer = Some(download);
            api.PJRT_Client_BufferFromHostBuffer = Some(upload);
            // Probe/table outlive every owner; lifetime-table destroy stubs do
            // not dereference synthetic handles. No native plugin is loaded.
            let raw_token: *mut PJRT_Client = ptr::from_ref(&probe).cast_mut().cast();
            let device_token: *mut PJRT_Device = ptr::from_ref(&probe).cast_mut().cast();
            let client = Client(Arc::new(ClientInner {
                plugin: Plugin(Arc::new(PluginInner {
                    api: &api,
                    _library: Some(ManuallyDrop::new(
                        libloading::os::unix::Library::this().into(),
                    )),
                })),
                raw: raw_token,
                device: device_token,
                addressable_devices: vec![device_token],
                _key_value_callbacks: None,
            }));
            assert!(matches!(
                BufferInner::new(client.0.clone(), ptr::null_mut()),
                Err(Error::InvalidPluginData { .. })
            ));
            let buffer = Buffer {
                inner: Arc::new(BufferInner {
                    client: client.0.clone(),
                    raw: NonNull::new(ptr::from_ref(&probe).cast_mut().cast()).unwrap(),
                }),
            };
            assert_eq!(buffer.host_payload_bytes().unwrap(), usize::MAX - 3);
            assert_eq!(probe.downloads.get(), 0);
            assert_eq!(probe.uploads.get(), 0);
            for limit in [0, 1024] {
                let failure = match buffer.copy_to_client_via_host_with_limit(&client, limit) {
                    Ok(_) => panic!("budget accepted impossible allocation"),
                    Err(error) => error,
                };
                assert!(matches!(
                    failure,
                    Error::LimitExceeded { message } if message.contains("exceeds limit")
                ));
            }
            // Even an unlimited copy must report impossible Vec capacity as an
            // ordinary error, without a capacity panic or payload operation.
            let failure = match buffer.copy_to_client_via_host(&client) {
                Ok(_) => panic!("accepted impossible allocation"),
                Err(error) => error,
            };
            assert!(matches!(
                failure,
                Error::ResourceExhausted {
                    operation: "host download allocation",
                    ..
                }
            ));
            assert_eq!(probe.queries.get(), 4);
            assert_eq!(probe.downloads.get(), 0);
            assert_eq!(probe.uploads.get(), 0);
        }
    }

    #[test]
    fn executable_placement_rejects_unsupported_device_lists() {
        struct Placement {
            devices: [*mut PJRT_Device; 2],
            count: usize,
            null_list: bool,
        }
        unsafe extern "C" fn devices(
            a: *mut PJRT_LoadedExecutable_AddressableDevices_Args,
        ) -> *mut PJRT_Error {
            let a = unsafe { &mut *a };
            let placement = unsafe { &*a.executable.cast::<Placement>() };
            a.num_addressable_devices = placement.count;
            a.addressable_devices = if placement.null_list {
                ptr::null()
            } else {
                placement.devices.as_ptr()
            };
            ptr::null_mut()
        }
        let selected_token = 0u8;
        let foreign_token = 1u8;
        let selected = ptr::from_ref(&selected_token).cast_mut().cast();
        let foreign = ptr::from_ref(&foreign_token).cast_mut().cast();
        let mut api = lifetime_table();
        api.PJRT_LoadedExecutable_AddressableDevices = Some(devices);
        let plugin = Plugin(Arc::new(PluginInner {
            api: &api,
            _library: Some(ManuallyDrop::new(
                libloading::os::unix::Library::this().into(),
            )),
        }));
        for (count, device, null_list, valid) in [
            (0, selected, false, false),
            (2, selected, false, true),
            (1, foreign, false, true),
            (1, ptr::null_mut(), false, false),
            (1, selected, true, false),
            (1, selected, false, true),
        ] {
            let placement = Placement {
                devices: [device, foreign],
                count,
                null_list,
            };
            let client = ClientInner {
                plugin: plugin.clone(),
                raw: ptr::null_mut(),
                device: selected,
                addressable_devices: vec![selected, foreign],
                _key_value_callbacks: None,
            };
            let result = executable_devices(&client, ptr::from_ref(&placement).cast_mut().cast());
            assert_eq!(result.is_ok(), valid);
        }
    }

    fn lifetime_table() -> PJRT_Api {
        // A synthetic, non-executed table: only lifetime capabilities are present.
        let mut api: PJRT_Api = unsafe { std::mem::zeroed() };
        api.struct_size = std::mem::size_of::<PJRT_Api>();
        unsafe extern "C" fn message(_: *mut PJRT_Error_Message_Args) {}
        unsafe extern "C" fn destroy(_: *mut PJRT_Error_Destroy_Args) {}
        api.PJRT_Error_Message = Some(message);
        api.PJRT_Error_Destroy = Some(destroy);
        macro_rules! install {
            ($name:ident, $args:ident) => {{
                unsafe extern "C" fn stub(_: *mut $args) -> *mut PJRT_Error {
                    ptr::null_mut()
                }
                api.$name = Some(stub);
            }};
        }
        install!(PJRT_Error_GetCode, PJRT_Error_GetCode_Args);
        install!(PJRT_Event_Await, PJRT_Event_Await_Args);
        install!(PJRT_Event_Destroy, PJRT_Event_Destroy_Args);
        install!(PJRT_Client_Destroy, PJRT_Client_Destroy_Args);
        install!(PJRT_Buffer_Destroy, PJRT_Buffer_Destroy_Args);
        install!(
            PJRT_LoadedExecutable_Destroy,
            PJRT_LoadedExecutable_Destroy_Args
        );
        install!(PJRT_Executable_Destroy, PJRT_Executable_Destroy_Args);
        api
    }

    #[test]
    fn lifecycle_preflight_rejects_each_missing_capability() {
        assert!(validate_lifetime_api(&lifetime_table()).is_ok());
        macro_rules! missing {
            ($name:ident) => {{
                let mut api = lifetime_table();
                api.$name = None;
                assert!(matches!(
                    validate_lifetime_api(&api).unwrap_err(),
                    Error::IncompatiblePlugin { message }
                        if message == concat!("null API slot: ", stringify!($name))
                ));
            }};
        }
        missing!(PJRT_Error_Message);
        missing!(PJRT_Error_GetCode);
        missing!(PJRT_Error_Destroy);
        missing!(PJRT_Event_Await);
        missing!(PJRT_Event_Destroy);
        missing!(PJRT_Client_Destroy);
        missing!(PJRT_Buffer_Destroy);
        missing!(PJRT_LoadedExecutable_Destroy);
        missing!(PJRT_Executable_Destroy);
    }

    #[test]
    fn lifecycle_preflight_respects_table_size() {
        let mut api = lifetime_table();
        api.struct_size = std::mem::offset_of!(PJRT_Api, PJRT_Error_Message);
        assert!(matches!(
            validate_lifetime_api(&api).unwrap_err(),
            Error::IncompatiblePlugin { message }
                if message == "missing API slot: PJRT_Error_Message"
        ));
    }

    #[test]
    fn collectives_extension_capability_is_discovered_without_assuming_support() {
        let mut tail = PJRT_Extension_Base {
            struct_size: PJRT_Extension_Base_STRUCT_SIZE as usize,
            type_: PJRT_Extension_Type_PJRT_Extension_Type_Layouts,
            next: ptr::null_mut(),
        };
        let mut collectives = PJRT_Extension_Base {
            struct_size: PJRT_Extension_Base_STRUCT_SIZE as usize,
            type_: PJRT_Extension_Type_PJRT_Extension_Type_Collectives,
            next: ptr::from_mut(&mut tail),
        };
        let mut api = lifetime_table();
        let plugin = |api: &PJRT_Api| {
            Plugin(Arc::new(PluginInner {
                api,
                _library: Some(ManuallyDrop::new(
                    libloading::os::unix::Library::this().into(),
                )),
            }))
        };
        assert!(!plugin(&api).supports_collectives_extension());
        api.extension_start = ptr::from_mut(&mut collectives);
        assert!(plugin(&api).supports_collectives_extension());
    }

    #[test]
    fn gpu_custom_call_extension_registers_typed_handlers() {
        use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);
        static API_VERSION: AtomicI32 = AtomicI32::new(-1);
        static NAME_SIZE: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn register(args: *mut PjrtGpuRegisterCustomCallArgs) -> *mut PJRT_Error {
            let args = unsafe { &*args };
            CALLS.fetch_add(1, Ordering::Relaxed);
            API_VERSION.store(args.api_version, Ordering::Relaxed);
            NAME_SIZE.store(args.function_name_size, Ordering::Relaxed);
            ptr::null_mut()
        }

        let mut extension = PjrtGpuCustomCallExtension {
            base: PJRT_Extension_Base {
                struct_size: std::mem::size_of::<PjrtGpuCustomCallExtension>(),
                type_: PJRT_Extension_Type_PJRT_Extension_Type_Gpu_Custom_Call,
                next: ptr::null_mut(),
            },
            custom_call: Some(register),
        };
        let mut api = lifetime_table();
        api.extension_start = ptr::from_mut(&mut extension).cast();
        let plugin = Plugin(Arc::new(PluginInner {
            api: &api,
            _library: Some(ManuallyDrop::new(
                libloading::os::unix::Library::this().into(),
            )),
        }));
        let execute = NonNull::<u8>::dangling().cast();
        assert!(plugin.supports_gpu_custom_calls());
        unsafe {
            plugin
                .register_gpu_custom_call(
                    "rxla.test",
                    GpuCustomCallApi::TypedFfi,
                    GpuCustomCallHandlers::execute_only(execute),
                )
                .unwrap();
        }
        assert_eq!(CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(API_VERSION.load(Ordering::Relaxed), 1);
        assert_eq!(NAME_SIZE.load(Ordering::Relaxed), "rxla.test".len());
    }

    #[test]
    fn dma_mapping_capability_requires_both_api_slots() {
        let mut api = lifetime_table();
        let plugin = |api: &PJRT_Api| {
            Plugin(Arc::new(PluginInner {
                api,
                _library: Some(ManuallyDrop::new(
                    libloading::os::unix::Library::this().into(),
                )),
            }))
        };
        assert!(!plugin(&api).has_dma_mapping_api());
        macro_rules! install {
            ($name:ident, $args:ident) => {{
                unsafe extern "C" fn stub(_: *mut $args) -> *mut PJRT_Error {
                    ptr::null_mut()
                }
                api.$name = Some(stub);
            }};
        }
        install!(PJRT_Client_DmaMap, PJRT_Client_DmaMap_Args);
        assert!(!plugin(&api).has_dma_mapping_api());
        install!(PJRT_Client_DmaUnmap, PJRT_Client_DmaUnmap_Args);
        assert!(plugin(&api).has_dma_mapping_api());
        api.struct_size = std::mem::offset_of!(PJRT_Api, PJRT_Client_DmaUnmap);
        assert!(!plugin(&api).has_dma_mapping_api());
    }

    #[test]
    fn compiler_program_preserves_open_format_labels() {
        let code = [1, 2, 3];
        assert_eq!(Program::hlo(&code), Program::new("hlo", &code));
        assert_eq!(Program::mlir(&code), Program::new("mlir", &code));
        assert_eq!(Program::new("vendor_ir", &code).format(), "vendor_ir");
        assert_eq!(Program::new("vendor_ir", &code).code(), code);
    }
}
