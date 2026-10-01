//! Opt-in CUDA profiler range with checked start/stop and RAII error cleanup.

use libloading::Library;
use snafu::{ResultExt, Snafu, ensure};
use std::path::Path;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Cannot load CUDA profiler runtime: {source}"))]
    Load { source: libloading::Error },
    #[snafu(display("Cannot resolve CUDA profiler API: {source}"))]
    Symbol { source: libloading::Error },
    #[snafu(display("CUDA profiler {operation} failed with status {status}"))]
    Status {
        operation: &'static str,
        status: i32,
    },
}

pub struct CudaProfiler {
    _library: Library,
    stop: unsafe extern "C" fn() -> i32,
    active: bool,
}

impl CudaProfiler {
    /// Loads a caller-selected trusted CUDA runtime with its stable C ABI.
    pub fn start(path: &Path) -> Result<Self, Error> {
        // SAFETY: CUDA runtime symbols have these signatures; the library is
        // retained until both explicit stop and any error cleanup have finished.
        let library = unsafe { Library::new(path) }.context(LoadSnafu)?;
        let start = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> i32>(b"cudaProfilerStart\0")
                .context(SymbolSnafu)?
        };
        let stop = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> i32>(b"cudaProfilerStop\0")
                .context(SymbolSnafu)?
        };
        // SAFETY: function pointer is valid while `library` is alive.
        let status = unsafe { start() };
        ensure!(
            status == 0,
            StatusSnafu {
                operation: "start",
                status
            }
        );
        Ok(Self {
            _library: library,
            stop,
            active: true,
        })
    }

    pub fn finish(mut self) -> Result<(), Error> {
        // SAFETY: retained library keeps the symbol alive.
        let status = unsafe { (self.stop)() };
        self.active = false;
        ensure!(
            status == 0,
            StatusSnafu {
                operation: "stop",
                status
            }
        );
        Ok(())
    }
}

impl Drop for CudaProfiler {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: retained library keeps the symbol alive during unwinding.
            let _ = unsafe { (self.stop)() };
        }
    }
}
