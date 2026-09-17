use super::*;
use snafu::{OptionExt, ensure};

/// Owned diagnostic metadata; no borrowed plugin strings or native handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: i32,
    pub kind: String,
    pub description: String,
    /// This is the device currently used for this client's buffer transfers.
    pub selected: bool,
}

/// Backend-owned memory space associated with a materialized PJRT buffer.
/// `kind` is intentionally an owned string: memory kinds are backend-defined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    pub id: i32,
    pub kind: String,
}

/// Descriptive metadata, not a sufficient persistent executable-cache fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientInfo {
    pub api_major: i32,
    pub api_minor: i32,
    pub platform: String,
    pub platform_version: String,
    pub process_index: i32,
    pub addressable_devices: Vec<DeviceInfo>,
}

// The caller guarantees a live PJRT-owned string for the supplied length.
unsafe fn copy_string(data: *const std::os::raw::c_char, len: usize) -> Result<String> {
    if len == 0 {
        return Ok(String::new());
    }
    ensure!(
        !data.is_null(),
        InvalidPluginDataSnafu {
            message: "null plugin string with nonzero length",
        }
    );
    ensure!(
        len <= isize::MAX as usize,
        InvalidPluginDataSnafu {
            message: "plugin string length exceeds addressable slice size",
        }
    );
    Ok(
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(data.cast(), len) })
            .into_owned(),
    )
}

impl Client {
    pub fn info(&self) -> Result<ClientInfo> {
        let plugin = &self.0.plugin;
        // load() validated that the complete version prefix exists.
        let version = unsafe { ptr::addr_of!((*plugin.api()).pjrt_api_version).read() };
        let platform = pjrt_call!(
            plugin,
            PJRT_Client_PlatformName,
            PJRT_Client_PlatformName_Args,
            PJRT_Client_PlatformName_Args_STRUCT_SIZE,
            client = self.0.raw,
        );
        let platform = unsafe { copy_string(platform.platform_name, platform.platform_name_size) }?;
        let build = pjrt_call!(
            plugin,
            PJRT_Client_PlatformVersion,
            PJRT_Client_PlatformVersion_Args,
            PJRT_Client_PlatformVersion_Args_STRUCT_SIZE,
            client = self.0.raw,
        );
        let platform_version =
            unsafe { copy_string(build.platform_version, build.platform_version_size) }?;
        let process = pjrt_call!(
            plugin,
            PJRT_Client_ProcessIndex,
            PJRT_Client_ProcessIndex_Args,
            PJRT_Client_ProcessIndex_Args_STRUCT_SIZE,
            client = self.0.raw,
        );
        let devices = pjrt_call!(
            plugin,
            PJRT_Client_AddressableDevices,
            PJRT_Client_AddressableDevices_Args,
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
            client = self.0.raw,
        );
        ensure!(
            devices.num_addressable_devices == 0 || !devices.addressable_devices.is_null(),
            InvalidPluginDataSnafu {
                message: "null addressable device list",
            }
        );
        let mut addressable_devices = Vec::with_capacity(devices.num_addressable_devices);
        for index in 0..devices.num_addressable_devices {
            let device = unsafe { *devices.addressable_devices.add(index) };
            ensure!(
                !device.is_null(),
                InvalidPluginDataSnafu {
                    message: "null addressable device",
                }
            );
            let description = pjrt_call!(
                plugin,
                PJRT_Device_GetDescription,
                PJRT_Device_GetDescription_Args,
                PJRT_Device_GetDescription_Args_STRUCT_SIZE,
                device = device,
            );
            ensure!(
                !description.device_description.is_null(),
                InvalidPluginDataSnafu {
                    message: "null device description",
                }
            );
            let id = pjrt_call!(
                plugin,
                PJRT_DeviceDescription_Id,
                PJRT_DeviceDescription_Id_Args,
                PJRT_DeviceDescription_Id_Args_STRUCT_SIZE,
                device_description = description.device_description,
            );
            let kind = pjrt_call!(
                plugin,
                PJRT_DeviceDescription_Kind,
                PJRT_DeviceDescription_Kind_Args,
                PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE,
                device_description = description.device_description,
            );
            let kind = unsafe { copy_string(kind.device_kind, kind.device_kind_size) }?;
            let text = pjrt_call!(
                plugin,
                PJRT_DeviceDescription_DebugString,
                PJRT_DeviceDescription_DebugString_Args,
                PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE,
                device_description = description.device_description,
            );
            addressable_devices.push(DeviceInfo {
                id: id.id,
                kind,
                description: unsafe { copy_string(text.debug_string, text.debug_string_size) }?,
                selected: device == self.0.device,
            });
        }
        Ok(ClientInfo {
            api_major: version.major_version,
            api_minor: version.minor_version,
            platform,
            platform_version,
            process_index: process.process_index,
            addressable_devices,
        })
    }
}

impl Buffer {
    /// Device on which this buffer is materialized.
    pub fn device_info(&self) -> Result<DeviceInfo> {
        let index = self.device_index()?;
        Client(self.inner.client.clone())
            .info()?
            .addressable_devices
            .into_iter()
            .nth(index)
            .context(InvalidPluginDataSnafu {
                message: "buffer device is absent from client metadata",
            })
    }

    /// Backend-defined memory space containing this buffer.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let plugin = &self.inner.client.plugin;
        let memory = pjrt_call!(
            plugin,
            PJRT_Buffer_Memory,
            PJRT_Buffer_Memory_Args,
            PJRT_Buffer_Memory_Args_STRUCT_SIZE,
            buffer = self.inner.raw.as_ptr(),
        );
        ensure!(
            !memory.memory.is_null(),
            InvalidPluginDataSnafu {
                message: "buffer has null PJRT memory",
            }
        );
        let id = pjrt_call!(
            plugin,
            PJRT_Memory_Id,
            PJRT_Memory_Id_Args,
            PJRT_Memory_Id_Args_STRUCT_SIZE,
            memory = memory.memory,
        );
        let kind = pjrt_call!(
            plugin,
            PJRT_Memory_Kind,
            PJRT_Memory_Kind_Args,
            PJRT_Memory_Kind_Args_STRUCT_SIZE,
            memory = memory.memory,
        );
        Ok(MemoryInfo {
            id: id.id,
            kind: unsafe { copy_string(kind.kind, kind.kind_size) }?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plugin_strings_are_length_delimited_and_owned() {
        assert_eq!(unsafe { copy_string(ptr::null(), 0) }.unwrap(), "");
        assert!(unsafe { copy_string(ptr::null(), 1) }.is_err());
        let mut bytes = *b"abcXYZ";
        assert!(unsafe { copy_string(bytes.as_ptr().cast(), usize::MAX) }.is_err());
        let string = unsafe { copy_string(bytes.as_ptr().cast(), 3) }.unwrap();
        bytes.fill(0);
        assert_eq!(string, "abc");
    }
}
