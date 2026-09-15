//! Checked-in OpenXLA protobuf definitions. Regenerate with xtask, not build.rs.
#![allow(clippy::all)]
pub mod xla {
    include!("generated/xla.rs");
    pub mod autotuner {
        include!("generated/xla.autotuner.rs");
    }
    pub mod buffer_assignment {
        include!("generated/xla.buffer_assignment.rs");
    }
    pub mod cpu {
        include!("generated/xla.cpu.rs");
    }
}
pub mod stream_executor {
    include!("generated/stream_executor.rs");
    pub mod dnn {
        include!("generated/stream_executor.dnn.rs");
    }
}
