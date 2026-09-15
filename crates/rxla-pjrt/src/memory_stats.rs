use super::*;

/// Owned allocator diagnostics for the client's selected device.
/// Optional counters remain None when the backend does not report them.
/// These are backend allocator counters, not process RSS or total board VRAM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMemoryStats {
    pub bytes_in_use: i64,
    pub peak_bytes_in_use: Option<i64>,
    pub num_allocs: Option<i64>,
    pub largest_alloc_size: Option<i64>,
    pub bytes_limit: Option<i64>,
    pub bytes_reserved: Option<i64>,
    pub peak_bytes_reserved: Option<i64>,
    pub bytes_reservable_limit: Option<i64>,
    pub largest_free_block_bytes: Option<i64>,
    pub pool_bytes: Option<i64>,
    pub peak_pool_bytes: Option<i64>,
    pub peak_allocated_bytes: Option<i64>,
}

impl From<PJRT_Device_MemoryStats_Args> for DeviceMemoryStats {
    fn from(a: PJRT_Device_MemoryStats_Args) -> Self {
        Self {
            bytes_in_use: a.bytes_in_use,
            peak_bytes_in_use: a.peak_bytes_in_use_is_set.then_some(a.peak_bytes_in_use),
            num_allocs: a.num_allocs_is_set.then_some(a.num_allocs),
            largest_alloc_size: a.largest_alloc_size_is_set.then_some(a.largest_alloc_size),
            bytes_limit: a.bytes_limit_is_set.then_some(a.bytes_limit),
            bytes_reserved: a.bytes_reserved_is_set.then_some(a.bytes_reserved),
            peak_bytes_reserved: a
                .peak_bytes_reserved_is_set
                .then_some(a.peak_bytes_reserved),
            bytes_reservable_limit: a
                .bytes_reservable_limit_is_set
                .then_some(a.bytes_reservable_limit),
            largest_free_block_bytes: a
                .largest_free_block_bytes_is_set
                .then_some(a.largest_free_block_bytes),
            pool_bytes: a.pool_bytes_is_set.then_some(a.pool_bytes),
            peak_pool_bytes: a.peak_pool_bytes_is_set.then_some(a.peak_pool_bytes),
            peak_allocated_bytes: a
                .peak_allocated_bytes_is_set
                .then_some(a.peak_allocated_bytes),
        }
    }
}

impl Client {
    /// Query diagnostic allocator counters on this client's selected device.
    /// Missing API slots and unsupported backends return errors, not zero usage.
    /// This does not synchronize outstanding work or reset peak counters; call
    /// after the operations whose completed memory footprint you want to inspect.
    /// Backend-defined peaks may include compilation/warmup and shared activity.
    pub fn memory_stats(&self) -> Result<DeviceMemoryStats> {
        let plugin = &self.0.plugin;
        let a = pjrt_call!(
            plugin,
            PJRT_Device_MemoryStats,
            PJRT_Device_MemoryStats_Args,
            PJRT_Device_MemoryStats_Args_STRUCT_SIZE,
            device = self.0.device,
        );
        Ok(a.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optional_counters_preserve_absence_and_zero() {
        let mut a = args!(
            PJRT_Device_MemoryStats_Args,
            PJRT_Device_MemoryStats_Args_STRUCT_SIZE
        );
        a.bytes_in_use = 42;
        a.peak_bytes_in_use = 123;
        let absent = DeviceMemoryStats::from(a);
        assert_eq!(absent.bytes_in_use, 42);
        assert_eq!(absent.peak_bytes_in_use, None);
        a.peak_bytes_in_use_is_set = true;
        a.peak_bytes_in_use = 0;
        a.num_allocs_is_set = true;
        a.num_allocs = 0;
        a.largest_alloc_size_is_set = true;
        a.largest_alloc_size = 0;
        a.bytes_limit_is_set = true;
        a.bytes_limit = 0;
        a.bytes_reserved_is_set = true;
        a.bytes_reserved = 0;
        a.peak_bytes_reserved_is_set = true;
        a.peak_bytes_reserved = 0;
        a.bytes_reservable_limit_is_set = true;
        a.bytes_reservable_limit = 0;
        a.largest_free_block_bytes_is_set = true;
        a.largest_free_block_bytes = 0;
        a.pool_bytes_is_set = true;
        a.pool_bytes = 0;
        a.peak_pool_bytes_is_set = true;
        a.peak_pool_bytes = 0;
        a.peak_allocated_bytes_is_set = true;
        a.peak_allocated_bytes = 0;
        let present = DeviceMemoryStats::from(a);
        assert_eq!(present.peak_bytes_in_use, Some(0));
        assert_eq!(present.num_allocs, Some(0));
        assert_eq!(present.largest_alloc_size, Some(0));
        assert_eq!(present.bytes_limit, Some(0));
        assert_eq!(present.bytes_reserved, Some(0));
        assert_eq!(present.peak_bytes_reserved, Some(0));
        assert_eq!(present.bytes_reservable_limit, Some(0));
        assert_eq!(present.largest_free_block_bytes, Some(0));
        assert_eq!(present.pool_bytes, Some(0));
        assert_eq!(present.peak_pool_bytes, Some(0));
        assert_eq!(present.peak_allocated_bytes, Some(0));
    }
}
