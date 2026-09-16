//! PJRT C API bindings and owned runtime handles. No native build at Cargo time.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
pub mod sys {
    include!("generated.rs");
}
mod distributed;
mod runtime;
pub use distributed::{
    DistributedClientConfig, InMemoryKeyValueStore, KeyValueStore, KeyValueStoreError,
};
pub use half::{bf16, f16};
pub use runtime::{
    Buffer, BufferMemoryLayout, Client, ClientInfo, ClientOptionValue, ClientOptions, DType,
    DeviceInfo, DeviceMemoryStats, Element, Error, Executable, OptimizedProgram, PendingExecution,
    PjrtErrorCode, Plugin, PluginRegistry, Program,
};
pub use runtime::{ByteStrides, HostView, Shape, StridedLayout};
