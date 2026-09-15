use super::*;
use snafu::ensure;

#[derive(Clone, Debug, PartialEq)]
enum StoredValue {
    String(String),
    Int64(i64),
    Int64List(Vec<i64>),
    Float(f32),
    Bool(bool),
}

/// Plugin-specific options used while creating a PJRT client.
///
/// Values are ordinary Rust values; [`ClientOptions::set`] owns any backing
/// storage that the synchronous PJRT create call needs.
///
/// ```
/// use rxla_pjrt::ClientOptions;
///
/// let options = ClientOptions::new()
///     .set("allocator", "bfc")
///     .set("preallocate", false)
///     .set("memory_fraction", 0.25_f32)
///     .set("device_ids", [0_i64, 1]);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClientOptions {
    entries: Vec<(String, StoredValue)>,
}

impl ClientOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a plugin-specific option, replacing an earlier value with the same name.
    pub fn set(mut self, name: impl Into<String>, value: impl ClientOptionValue) -> Self {
        value.add_to_client_options(name.into(), &mut self);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn insert(&mut self, name: String, value: StoredValue) {
        if let Some((_, current)) = self.entries.iter_mut().find(|(key, _)| key == &name) {
            *current = value;
        } else {
            self.entries.push((name, value));
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust value accepted by [`ClientOptions::set`].
///
/// This trait is sealed because the PJRT C API supports a closed set of option
/// representations.
pub trait ClientOptionValue: sealed::Sealed {
    #[doc(hidden)]
    fn add_to_client_options(self, name: String, options: &mut ClientOptions);
}

macro_rules! scalar_option {
    ($ty:ty, $variant:ident) => {
        impl sealed::Sealed for $ty {}
        impl ClientOptionValue for $ty {
            fn add_to_client_options(self, name: String, options: &mut ClientOptions) {
                options.insert(name, StoredValue::$variant(self));
            }
        }
    };
}

scalar_option!(i64, Int64);
scalar_option!(f32, Float);
scalar_option!(bool, Bool);

impl sealed::Sealed for String {}
impl ClientOptionValue for String {
    fn add_to_client_options(self, name: String, options: &mut ClientOptions) {
        options.insert(name, StoredValue::String(self));
    }
}

impl sealed::Sealed for &str {}
impl ClientOptionValue for &str {
    fn add_to_client_options(self, name: String, options: &mut ClientOptions) {
        options.insert(name, StoredValue::String(self.into()));
    }
}

impl sealed::Sealed for Vec<i64> {}
impl ClientOptionValue for Vec<i64> {
    fn add_to_client_options(self, name: String, options: &mut ClientOptions) {
        options.insert(name, StoredValue::Int64List(self));
    }
}

impl sealed::Sealed for &[i64] {}
impl ClientOptionValue for &[i64] {
    fn add_to_client_options(self, name: String, options: &mut ClientOptions) {
        options.insert(name, StoredValue::Int64List(self.into()));
    }
}

impl<const N: usize> sealed::Sealed for [i64; N] {}
impl<const N: usize> ClientOptionValue for [i64; N] {
    fn add_to_client_options(self, name: String, options: &mut ClientOptions) {
        options.insert(name, StoredValue::Int64List(self.into()));
    }
}

impl ClientOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.entries.iter().all(|(name, _)| !name.is_empty()),
            InvalidArgumentSnafu {
                message: "client option names must be nonempty",
            }
        );
        Ok(())
    }

    // Returned pointers borrow `self`; only the synchronous create call may use
    // them. Neither the input values nor the strings are mutated during that call.
    pub(super) fn encode(&self) -> Result<Vec<PJRT_NamedValue>> {
        self.validate()?;
        self.entries
            .iter()
            .map(|(name, value)| {
                let mut raw = args!(PJRT_NamedValue, PJRT_NamedValue_STRUCT_SIZE);
                raw.name = name.as_ptr().cast();
                raw.name_size = name.len();
                raw.value_size = 1;
                match value {
                    StoredValue::String(v) => {
                        raw.type_ = PJRT_NamedValue_Type_PJRT_NamedValue_kString;
                        raw.__bindgen_anon_1.string_value = v.as_ptr().cast();
                        raw.value_size = v.len();
                    }
                    StoredValue::Int64(v) => {
                        raw.type_ = PJRT_NamedValue_Type_PJRT_NamedValue_kInt64;
                        raw.__bindgen_anon_1.int64_value = *v;
                    }
                    StoredValue::Int64List(v) => {
                        raw.type_ = PJRT_NamedValue_Type_PJRT_NamedValue_kInt64List;
                        raw.__bindgen_anon_1.int64_array_value = v.as_ptr();
                        raw.value_size = v.len();
                    }
                    StoredValue::Float(v) => {
                        raw.type_ = PJRT_NamedValue_Type_PJRT_NamedValue_kFloat;
                        raw.__bindgen_anon_1.float_value = *v;
                    }
                    StoredValue::Bool(v) => {
                        raw.type_ = PJRT_NamedValue_Type_PJRT_NamedValue_kBool;
                        raw.__bindgen_anon_1.bool_value = *v;
                    }
                }
                Ok(raw)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_encodes_all_rust_value_types() {
        let options = ClientOptions::new()
            .set("a", "bfc")
            .set("b", -9_i64)
            .set("c", [0_i64, 2])
            .set("d", 0.25_f32)
            .set("e", false)
            .set("f", String::new())
            .set("g", Vec::<i64>::new());
        let raw = options.encode().unwrap();
        for (i, r) in raw.iter().enumerate() {
            assert_eq!(r.struct_size, PJRT_NamedValue_STRUCT_SIZE as usize);
            assert!(r.extension_start.is_null());
            assert_eq!(r.name, options.entries[i].0.as_ptr().cast());
            assert_eq!(r.name_size, 1);
        }
        assert_eq!(
            raw.iter().map(|r| r.type_).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 0, 2]
        );
        assert_eq!(
            raw.iter().map(|r| r.value_size).collect::<Vec<_>>(),
            [3, 1, 2, 1, 1, 0, 0]
        );
        unsafe {
            assert_eq!(
                std::slice::from_raw_parts(raw[0].__bindgen_anon_1.string_value.cast::<u8>(), 3),
                b"bfc"
            );
            assert_eq!(raw[1].__bindgen_anon_1.int64_value, -9);
            assert_eq!(
                std::slice::from_raw_parts(raw[2].__bindgen_anon_1.int64_array_value, 2),
                [0, 2]
            );
            assert_eq!(raw[3].__bindgen_anon_1.float_value, 0.25);
            assert!(!raw[4].__bindgen_anon_1.bool_value);
        }
    }

    #[test]
    fn set_replaces_an_existing_name_without_reordering_it() {
        let options = ClientOptions::new().set("x", 1_i64).set("x", 2_i64);
        assert_eq!(options.entries.len(), 1);
        let raw = options.encode().unwrap();
        assert_eq!(unsafe { raw[0].__bindgen_anon_1.int64_value }, 2);
    }

    #[test]
    fn rejects_an_empty_name_before_loading() {
        let options = ClientOptions::new().set("", false);
        let err = unsafe { Client::load_with_options("/nonexistent/pjrt.so", &options) }
            .err()
            .unwrap();
        assert!(err.to_string().contains("nonempty"));
    }
}
