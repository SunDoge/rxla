use super::*;
use snafu::{OptionExt, ensure};

/// Owned physical layout reported by the legacy PJRT layout query.
/// Not a view, allocation size, host-download layout or zero-copy guarantee.
/// Tiling dimensions are preserved verbatim (including backend sentinel values).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferMemoryLayout {
    Tiled {
        minor_to_major: Vec<i64>,
        tiles: Vec<Vec<i64>>,
    },
    Strided {
        byte_strides: ByteStrides,
    },
}

// Diagnostic metadata only; avoid unbounded allocation from corrupt lengths.
const MAX_METADATA_ELEMENTS: usize = 4096;

// The trusted plugin must supply readable, initialized memory. Checks here
// catch structural mistakes, not malicious pointers from native code.
unsafe fn copy_metadata<T: Copy>(pointer: *const T, len: usize) -> Result<Vec<T>> {
    ensure!(
        len <= MAX_METADATA_ELEMENTS,
        InvalidPluginDataSnafu {
            message: "layout metadata exceeds diagnostic limit",
        }
    );
    if len == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        !pointer.is_null() && (pointer as usize).is_multiple_of(std::mem::align_of::<T>()),
        InvalidPluginDataSnafu {
            message: "null or unaligned layout metadata",
        }
    );
    Ok(unsafe { std::slice::from_raw_parts(pointer, len) }.to_vec())
}

unsafe fn decode(
    layout: *const PJRT_Buffer_MemoryLayout,
    rank: usize,
) -> Result<BufferMemoryLayout> {
    // Upstream ConvertToBufferMemoryLayoutData does not initialize output
    // struct_size/extension fields. Do not read them, or copy the whole union
    // member into a Rust value. The outer query args size defines this ABI.
    // Read only the active payload fields that the native function initializes.
    match unsafe { ptr::addr_of!((*layout).type_).read() } {
        kind if kind == PJRT_Buffer_MemoryLayout_Type_PJRT_Buffer_MemoryLayout_Type_Tiled => {
            let tiled = unsafe { ptr::addr_of!((*layout).__bindgen_anon_1.tiled) };
            ensure!(
                unsafe { ptr::addr_of!((*tiled).minor_to_major_size).read() } == rank,
                InvalidPluginDataSnafu {
                    message: "invalid tiled layout structure or rank",
                }
            );
            let order =
                unsafe { copy_metadata(ptr::addr_of!((*tiled).minor_to_major).read(), rank) }?;
            let mut seen = vec![false; order.len()];
            for &axis in &order {
                let axis = usize::try_from(axis).ok().context(InvalidPluginDataSnafu {
                    message: "negative layout axis",
                })?;
                ensure!(
                    axis < rank && !seen[axis],
                    InvalidPluginDataSnafu {
                        message: "layout axes are not a permutation",
                    }
                );
                seen[axis] = true;
            }
            let num_tiles = unsafe { ptr::addr_of!((*tiled).num_tiles).read() };
            let sizes = if num_tiles == 0 {
                Vec::new()
            } else {
                unsafe { copy_metadata(ptr::addr_of!((*tiled).tile_dim_sizes).read(), num_tiles) }?
            };
            let total = sizes
                .iter()
                .try_fold(0usize, |sum, &n| sum.checked_add(n))
                .context(InvalidPluginDataSnafu {
                    message: "tile metadata size overflow",
                })?;
            let dims = if total == 0 {
                Vec::new()
            } else {
                unsafe { copy_metadata(ptr::addr_of!((*tiled).tile_dims).read(), total) }?
            };
            let mut offset = 0;
            let tiles = sizes
                .into_iter()
                .map(|n| {
                    let tile = dims[offset..offset + n].to_vec();
                    offset += n;
                    tile
                })
                .collect();
            Ok(BufferMemoryLayout::Tiled {
                minor_to_major: order,
                tiles,
            })
        }
        kind if kind == PJRT_Buffer_MemoryLayout_Type_PJRT_Buffer_MemoryLayout_Type_Strides => {
            let strides = unsafe { ptr::addr_of!((*layout).__bindgen_anon_1.strides) };
            ensure!(
                unsafe { ptr::addr_of!((*strides).num_byte_strides).read() } == rank,
                InvalidPluginDataSnafu {
                    message: "invalid strided layout structure or rank",
                }
            );
            Ok(BufferMemoryLayout::Strided {
                byte_strides: ByteStrides::new(&unsafe {
                    copy_metadata(ptr::addr_of!((*strides).byte_strides).read(), rank)
                }?),
            })
        }
        other => UnsupportedSnafu {
            message: format!("memory layout kind {other}"),
        }
        .fail(),
    }
}

fn query_function(
    api: *const PJRT_Api,
) -> Result<unsafe extern "C" fn(*mut PJRT_Buffer_GetMemoryLayout_Args) -> *mut PJRT_Error> {
    Ok(function!(api, PJRT_Buffer_GetMemoryLayout))
}

impl Buffer {
    /// Inspect physical storage without downloading data or waiting on execution.
    /// Copies buffer-owned metadata; the result survives buffer destruction.
    /// Uses the deprecated query in our pinned ABI, not the newer layout extension.
    /// Missing/unsupported queries and unknown layouts return errors, never dense
    /// defaults. This diagnostic does not prove readiness or create a native view.
    pub fn memory_layout(&self) -> Result<BufferMemoryLayout> {
        let plugin = &self.inner.client.plugin;
        let query = query_function(plugin.api())?;
        let rank = self.dimensions()?.len();
        let mut a = args!(
            PJRT_Buffer_GetMemoryLayout_Args,
            PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE
        );
        a.buffer = self.inner.raw.as_ptr();
        plugin.check(unsafe { query(&mut a) })?;
        // Header contract: all layout arrays live as long as self's buffer.
        unsafe { decode(ptr::addr_of!(a.layout), rank) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiled(order: &[i64], sizes: &[usize], dims: &[i64]) -> PJRT_Buffer_MemoryLayout {
        let mut layout = args!(
            PJRT_Buffer_MemoryLayout,
            PJRT_Buffer_MemoryLayout_STRUCT_SIZE
        );
        let mut t = args!(
            PJRT_Buffer_MemoryLayout_Tiled,
            PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE
        );
        t.minor_to_major = order.as_ptr();
        t.minor_to_major_size = order.len();
        t.tile_dim_sizes = sizes.as_ptr();
        t.num_tiles = sizes.len();
        t.tile_dims = dims.as_ptr();
        layout.__bindgen_anon_1.tiled = t;
        layout
    }

    #[test]
    fn tiled_metadata_is_owned_and_preserves_tile_groups() {
        let owned = {
            let order = vec![1, 0];
            let sizes = vec![2, 1];
            let dims = vec![8, 4, 2];
            unsafe { decode(&tiled(&order, &sizes, &dims), 2) }.unwrap()
        };
        assert_eq!(
            owned,
            BufferMemoryLayout::Tiled {
                minor_to_major: vec![1, 0],
                tiles: vec![vec![8, 4], vec![2]],
            }
        );
        assert_eq!(
            unsafe { decode(&tiled(&[], &[], &[]), 0) }.unwrap(),
            BufferMemoryLayout::Tiled {
                minor_to_major: vec![],
                tiles: vec![]
            }
        );
    }

    #[test]
    fn preserves_signed_and_broadcast_byte_strides() {
        let mut layout = args!(
            PJRT_Buffer_MemoryLayout,
            PJRT_Buffer_MemoryLayout_STRUCT_SIZE
        );
        layout.type_ = PJRT_Buffer_MemoryLayout_Type_PJRT_Buffer_MemoryLayout_Type_Strides;
        let mut s = args!(
            PJRT_Buffer_MemoryLayout_Strides,
            PJRT_Buffer_MemoryLayout_Strides_STRUCT_SIZE
        );
        let bytes = [-12, 0, 4];
        s.byte_strides = bytes.as_ptr();
        s.num_byte_strides = bytes.len();
        layout.__bindgen_anon_1.strides = s;
        assert_eq!(
            unsafe { decode(&layout, 3) }.unwrap(),
            BufferMemoryLayout::Strided {
                byte_strides: ByteStrides::new(&bytes)
            }
        );
        assert!(unsafe { decode(&layout, 2) }.is_err());
    }

    #[test]
    fn rejects_malformed_metadata_before_reading_payloads() {
        assert!(unsafe { decode(&tiled(&[0, 0], &[], &[]), 2) }.is_err());
        assert!(unsafe { decode(&tiled(&[-1], &[], &[]), 1) }.is_err());
        assert!(unsafe { decode(&tiled(&[2], &[], &[]), 1) }.is_err());
        let mut layout = tiled(&[], &[], &[]);
        layout.type_ = 99;
        assert!(unsafe { decode(&layout, 0) }.is_err());
        assert!(unsafe { copy_metadata::<i64>(ptr::null(), 1) }.is_err());
        assert!(unsafe { copy_metadata::<i64>(ptr::null(), 4097) }.is_err());
        assert!(unsafe { decode(&tiled(&[], &[usize::MAX, 1], &[]), 0) }.is_err());
    }

    #[test]
    fn legacy_output_headers_need_not_be_initialized() {
        let mut layout = std::mem::MaybeUninit::<PJRT_Buffer_MemoryLayout>::uninit();
        let p = layout.as_mut_ptr();
        let order = [1, 0];
        // Model the legacy upstream producer exactly: no headers, and no tile
        // pointers needed when there are no tiles. Never assume_init this value.
        unsafe {
            ptr::addr_of_mut!((*p).type_)
                .write(PJRT_Buffer_MemoryLayout_Type_PJRT_Buffer_MemoryLayout_Type_Tiled);
            let tiled = ptr::addr_of_mut!((*p).__bindgen_anon_1.tiled);
            ptr::addr_of_mut!((*tiled).minor_to_major).write(order.as_ptr());
            ptr::addr_of_mut!((*tiled).minor_to_major_size).write(2);
            ptr::addr_of_mut!((*tiled).num_tiles).write(0);
            assert_eq!(
                decode(p, 2).unwrap(),
                BufferMemoryLayout::Tiled {
                    minor_to_major: vec![1, 0],
                    tiles: vec![],
                }
            );
        }
    }

    #[test]
    fn missing_and_null_slots_are_not_dense_fallbacks() {
        let mut api: PJRT_Api = unsafe { std::mem::zeroed() };
        assert!(matches!(
            query_function(&api).unwrap_err(),
            Error::IncompatiblePlugin { message } if message.contains("missing API slot")
        ));
        api.struct_size = std::mem::size_of::<PJRT_Api>();
        assert!(matches!(
            query_function(&api).unwrap_err(),
            Error::IncompatiblePlugin { message } if message.contains("null API slot")
        ));
    }
}
