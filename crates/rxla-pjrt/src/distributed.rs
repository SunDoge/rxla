//! Safe callback bridge used by distributed PJRT clients.

use crate::{ClientOptions, Error, PjrtErrorCode, sys::*};
use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Error>;

/// Failure returned by a [`KeyValueStore`] implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyValueStoreError {
    code: PjrtErrorCode,
    message: String,
}

impl KeyValueStoreError {
    pub fn new(code: PjrtErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn code(&self) -> PjrtErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Process-shared rendezvous storage used while PJRT initializes distributed
/// devices and collective communication.
///
/// Implementations may use a coordination service, Redis, etcd, or another
/// durable service. Calls can arrive concurrently from PJRT background threads.
pub trait KeyValueStore: Send + Sync + 'static {
    /// Block until `key` exists or `timeout` expires.
    fn get(
        &self,
        key: &[u8],
        timeout: Duration,
    ) -> std::result::Result<Vec<u8>, KeyValueStoreError>;

    /// Return the current value without blocking.
    fn try_get(&self, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, KeyValueStoreError>;

    /// Publish a value. Keys and values are binary, not UTF-8 strings.
    fn put(&self, key: &[u8], value: &[u8]) -> std::result::Result<(), KeyValueStoreError>;
}

/// Validated topology and rendezvous configuration for a distributed client.
#[derive(Clone)]
pub struct DistributedClientConfig {
    node_id: usize,
    node_count: usize,
    store: Arc<dyn KeyValueStore>,
}

impl DistributedClientConfig {
    pub fn new(node_id: usize, node_count: usize, store: Arc<dyn KeyValueStore>) -> Result<Self> {
        if node_count == 0 {
            return Err(Error::InvalidArgument {
                message: "distributed node count must be nonzero".into(),
            });
        }
        if node_id >= node_count {
            return Err(Error::InvalidArgument {
                message: format!(
                    "distributed node id {node_id} is out of range for {node_count} nodes"
                ),
            });
        }
        if i64::try_from(node_id).is_err() || i64::try_from(node_count).is_err() {
            return Err(Error::LimitExceeded {
                message: "distributed topology does not fit the PJRT option representation".into(),
            });
        }
        Ok(Self {
            node_id,
            node_count,
            store,
        })
    }

    pub fn node_id(&self) -> usize {
        self.node_id
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub(super) fn options(&self, options: &ClientOptions) -> ClientOptions {
        options
            .clone()
            .set("node_id", self.node_id as i64)
            .set("num_nodes", self.node_count as i64)
    }

    pub(super) fn callbacks(&self) -> Box<KeyValueCallbackState> {
        Box::new(KeyValueCallbackState {
            store: self.store.clone(),
        })
    }
}

/// A thread-safe in-process rendezvous store, useful for tests and for multiple
/// PJRT clients hosted by one process. It is not a multi-process transport.
#[derive(Default)]
pub struct InMemoryKeyValueStore {
    values: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
    changed: Condvar,
}

impl KeyValueStore for InMemoryKeyValueStore {
    fn get(
        &self,
        key: &[u8],
        timeout: Duration,
    ) -> std::result::Result<Vec<u8>, KeyValueStoreError> {
        let deadline = Instant::now().checked_add(timeout);
        let mut values = self.values.lock().map_err(|_| store_poisoned())?;
        loop {
            if let Some(value) = values.get(key) {
                return Ok(value.clone());
            }
            let Some(deadline) = deadline else {
                return Err(KeyValueStoreError::new(
                    PjrtErrorCode::DeadlineExceeded,
                    "key-value get timeout is too large",
                ));
            };
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return Err(KeyValueStoreError::new(
                    PjrtErrorCode::DeadlineExceeded,
                    "key-value get timed out",
                ));
            };
            let (next, result) = self
                .changed
                .wait_timeout(values, remaining)
                .map_err(|_| store_poisoned())?;
            values = next;
            if result.timed_out() && !values.contains_key(key) {
                return Err(KeyValueStoreError::new(
                    PjrtErrorCode::DeadlineExceeded,
                    "key-value get timed out",
                ));
            }
        }
    }

    fn try_get(&self, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, KeyValueStoreError> {
        Ok(self
            .values
            .lock()
            .map_err(|_| store_poisoned())?
            .get(key)
            .cloned())
    }

    fn put(&self, key: &[u8], value: &[u8]) -> std::result::Result<(), KeyValueStoreError> {
        self.values
            .lock()
            .map_err(|_| store_poisoned())?
            .insert(key.to_vec(), value.to_vec());
        self.changed.notify_all();
        Ok(())
    }
}

fn store_poisoned() -> KeyValueStoreError {
    KeyValueStoreError::new(PjrtErrorCode::Internal, "key-value store lock poisoned")
}

pub(super) struct KeyValueCallbackState {
    store: Arc<dyn KeyValueStore>,
}

impl KeyValueCallbackState {
    pub(super) fn install(&mut self, args: &mut PJRT_Client_Create_Args) {
        let user_arg = ptr::from_mut(self).cast();
        args.kv_get_callback = Some(kv_get);
        args.kv_get_user_arg = user_arg;
        args.kv_try_get_callback = Some(kv_try_get);
        args.kv_try_get_user_arg = user_arg;
        args.kv_put_callback = Some(kv_put);
        args.kv_put_user_arg = user_arg;
    }
}

fn raw_code(code: PjrtErrorCode) -> PJRT_Error_Code {
    match code {
        PjrtErrorCode::Cancelled => PJRT_Error_Code_PJRT_Error_Code_CANCELLED,
        PjrtErrorCode::Unknown => PJRT_Error_Code_PJRT_Error_Code_UNKNOWN,
        PjrtErrorCode::InvalidArgument => PJRT_Error_Code_PJRT_Error_Code_INVALID_ARGUMENT,
        PjrtErrorCode::DeadlineExceeded => PJRT_Error_Code_PJRT_Error_Code_DEADLINE_EXCEEDED,
        PjrtErrorCode::NotFound => PJRT_Error_Code_PJRT_Error_Code_NOT_FOUND,
        PjrtErrorCode::AlreadyExists => PJRT_Error_Code_PJRT_Error_Code_ALREADY_EXISTS,
        PjrtErrorCode::PermissionDenied => PJRT_Error_Code_PJRT_Error_Code_PERMISSION_DENIED,
        PjrtErrorCode::ResourceExhausted => PJRT_Error_Code_PJRT_Error_Code_RESOURCE_EXHAUSTED,
        PjrtErrorCode::FailedPrecondition => PJRT_Error_Code_PJRT_Error_Code_FAILED_PRECONDITION,
        PjrtErrorCode::Aborted => PJRT_Error_Code_PJRT_Error_Code_ABORTED,
        PjrtErrorCode::OutOfRange => PJRT_Error_Code_PJRT_Error_Code_OUT_OF_RANGE,
        PjrtErrorCode::Unimplemented => PJRT_Error_Code_PJRT_Error_Code_UNIMPLEMENTED,
        PjrtErrorCode::Internal => PJRT_Error_Code_PJRT_Error_Code_INTERNAL,
        PjrtErrorCode::Unavailable => PJRT_Error_Code_PJRT_Error_Code_UNAVAILABLE,
        PjrtErrorCode::DataLoss => PJRT_Error_Code_PJRT_Error_Code_DATA_LOSS,
        PjrtErrorCode::Unauthenticated => PJRT_Error_Code_PJRT_Error_Code_UNAUTHENTICATED,
        PjrtErrorCode::Unrecognized(code) => code,
    }
}

unsafe fn callback_failure(
    callback: *mut PJRT_CallbackError,
    error: &KeyValueStoreError,
) -> *mut PJRT_Error {
    if callback.is_null() {
        return ptr::null_mut();
    }
    let Some(callback) = (unsafe { callback.read() }) else {
        return ptr::null_mut();
    };
    unsafe {
        callback(
            raw_code(error.code),
            error.message.as_ptr().cast(),
            error.message.len(),
        )
    }
}

unsafe fn callback_state<'a>(user_arg: *mut std::ffi::c_void) -> Option<&'a KeyValueCallbackState> {
    unsafe { user_arg.cast::<KeyValueCallbackState>().as_ref() }
}

unsafe fn callback_bytes<'a>(pointer: *const i8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        Some(&[])
    } else if pointer.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(pointer.cast(), len) })
    }
}

unsafe fn return_value(
    value: Vec<u8>,
    output: &mut *mut i8,
    output_size: &mut usize,
    deleter: &mut PJRT_KeyValueGetCallback_ValueDeleter,
) -> std::result::Result<(), KeyValueStoreError> {
    let allocation_size = value.len().max(1);
    let allocation = unsafe { libc::malloc(allocation_size) }.cast::<u8>();
    if allocation.is_null() {
        return Err(KeyValueStoreError::new(
            PjrtErrorCode::ResourceExhausted,
            "allocating PJRT key-value result failed",
        ));
    }
    unsafe { ptr::copy_nonoverlapping(value.as_ptr(), allocation, value.len()) };
    *output = allocation.cast();
    *output_size = value.len();
    *deleter = Some(free_value);
    Ok(())
}

unsafe extern "C" fn free_value(value: *mut i8) {
    unsafe { libc::free(value.cast()) };
}

unsafe extern "C" fn kv_get(args: *mut PJRT_KeyValueGetCallback_Args) -> *mut PJRT_Error {
    let Some(args) = (unsafe { args.as_mut() }) else {
        return ptr::null_mut();
    };
    let operation = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<(), KeyValueStoreError> {
            let state = unsafe { callback_state(args.user_arg) }.ok_or_else(|| {
                KeyValueStoreError::new(
                    PjrtErrorCode::InvalidArgument,
                    "null key-value callback state",
                )
            })?;
            let key = unsafe { callback_bytes(args.key, args.key_size) }.ok_or_else(|| {
                KeyValueStoreError::new(PjrtErrorCode::InvalidArgument, "null key-value key")
            })?;
            let timeout = u64::try_from(args.timeout_in_ms).map_err(|_| {
                KeyValueStoreError::new(
                    PjrtErrorCode::InvalidArgument,
                    "negative key-value timeout",
                )
            })?;
            let value = state.store.get(key, Duration::from_millis(timeout))?;
            unsafe {
                return_value(
                    value,
                    &mut args.value,
                    &mut args.value_size,
                    &mut args.value_deleter_callback,
                )
            }
        },
    ));
    match operation {
        Ok(Ok(())) => ptr::null_mut(),
        Ok(Err(error)) => unsafe { callback_failure(args.callback_error, &error) },
        Err(_) => unsafe {
            callback_failure(
                args.callback_error,
                &KeyValueStoreError::new(
                    PjrtErrorCode::Internal,
                    "key-value get callback panicked",
                ),
            )
        },
    }
}

unsafe extern "C" fn kv_try_get(args: *mut PJRT_KeyValueTryGetCallback_Args) -> *mut PJRT_Error {
    let Some(args) = (unsafe { args.as_mut() }) else {
        return ptr::null_mut();
    };
    let operation = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<(), KeyValueStoreError> {
            let state = unsafe { callback_state(args.user_arg) }.ok_or_else(|| {
                KeyValueStoreError::new(
                    PjrtErrorCode::InvalidArgument,
                    "null key-value callback state",
                )
            })?;
            let key = unsafe { callback_bytes(args.key, args.key_size) }.ok_or_else(|| {
                KeyValueStoreError::new(PjrtErrorCode::InvalidArgument, "null key-value key")
            })?;
            let value = state.store.try_get(key)?.ok_or_else(|| {
                KeyValueStoreError::new(PjrtErrorCode::NotFound, "key-value key was not found")
            })?;
            unsafe {
                return_value(
                    value,
                    &mut args.value,
                    &mut args.value_size,
                    &mut args.value_deleter_callback,
                )
            }
        },
    ));
    match operation {
        Ok(Ok(())) => ptr::null_mut(),
        Ok(Err(error)) => unsafe { callback_failure(args.callback_error, &error) },
        Err(_) => unsafe {
            callback_failure(
                args.callback_error,
                &KeyValueStoreError::new(
                    PjrtErrorCode::Internal,
                    "key-value try-get callback panicked",
                ),
            )
        },
    }
}

unsafe extern "C" fn kv_put(args: *mut PJRT_KeyValuePutCallback_Args) -> *mut PJRT_Error {
    let Some(args) = (unsafe { args.as_mut() }) else {
        return ptr::null_mut();
    };
    let operation = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<(), KeyValueStoreError> {
            let state = unsafe { callback_state(args.user_arg) }.ok_or_else(|| {
                KeyValueStoreError::new(
                    PjrtErrorCode::InvalidArgument,
                    "null key-value callback state",
                )
            })?;
            let key = unsafe { callback_bytes(args.key, args.key_size) }.ok_or_else(|| {
                KeyValueStoreError::new(PjrtErrorCode::InvalidArgument, "null key-value key")
            })?;
            let value =
                unsafe { callback_bytes(args.value, args.value_size) }.ok_or_else(|| {
                    KeyValueStoreError::new(PjrtErrorCode::InvalidArgument, "null key-value value")
                })?;
            state.store.put(key, value)
        },
    ));
    match operation {
        Ok(Ok(())) => ptr::null_mut(),
        Ok(Err(error)) => unsafe { callback_failure(args.callback_error, &error) },
        Err(_) => unsafe {
            callback_failure(
                args.callback_error,
                &KeyValueStoreError::new(
                    PjrtErrorCode::Internal,
                    "key-value put callback panicked",
                ),
            )
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{ptr, thread};

    #[test]
    fn in_memory_store_blocks_until_publish_and_times_out() {
        let store = Arc::new(InMemoryKeyValueStore::default());
        let reader = store.clone();
        let task = thread::spawn(move || reader.get(b"nccl-id", Duration::from_secs(1)));
        store.put(b"nccl-id", b"value").unwrap();
        assert_eq!(task.join().unwrap().unwrap(), b"value");
        assert_eq!(store.try_get(b"missing").unwrap(), None);
        assert_eq!(
            store
                .get(b"missing", Duration::from_millis(0))
                .unwrap_err()
                .code(),
            PjrtErrorCode::DeadlineExceeded
        );
    }

    #[test]
    fn callbacks_copy_binary_values_into_plugin_owned_results() {
        let store = Arc::new(InMemoryKeyValueStore::default());
        store.put(b"key", &[0, 1, 255]).unwrap();
        let mut state = KeyValueCallbackState { store };
        let mut args: PJRT_KeyValueGetCallback_Args = unsafe { std::mem::zeroed() };
        args.struct_size = PJRT_KeyValueGetCallback_Args_STRUCT_SIZE as usize;
        args.key = b"key".as_ptr().cast();
        args.key_size = 3;
        args.timeout_in_ms = 1;
        args.user_arg = ptr::from_mut(&mut state).cast();
        assert!(unsafe { kv_get(&mut args) }.is_null());
        assert_eq!(
            unsafe { std::slice::from_raw_parts(args.value.cast::<u8>(), args.value_size) },
            [0, 1, 255]
        );
        unsafe { args.value_deleter_callback.unwrap()(args.value) };
    }

    #[test]
    fn empty_callback_slices_never_form_a_slice_from_null() {
        assert_eq!(
            unsafe { callback_bytes(ptr::null(), 0) },
            Some([].as_slice())
        );
        assert_eq!(unsafe { callback_bytes(ptr::null(), 1) }, None);
    }

    #[test]
    fn distributed_config_validates_and_owns_topology_options() {
        let store = Arc::new(InMemoryKeyValueStore::default());
        assert!(DistributedClientConfig::new(0, 0, store.clone()).is_err());
        assert!(DistributedClientConfig::new(2, 2, store.clone()).is_err());
        let config = DistributedClientConfig::new(1, 2, store).unwrap();
        assert_eq!(config.node_id(), 1);
        assert_eq!(config.node_count(), 2);
        assert_eq!(
            config.options(&ClientOptions::new().set("allocator", "bfc")),
            ClientOptions::new()
                .set("allocator", "bfc")
                .set("node_id", 1_i64)
                .set("num_nodes", 2_i64)
        );
    }
}
