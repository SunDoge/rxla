//! The ergonomic entry point for RXLA applications.
//!
//! Core tensor and runtime types are available at the crate root. The facade
//! deliberately stays small:
//!
//! - [`nn`] contains parameter-effect model construction.
//! - [`ir`] and [`pjrt`] expose lower-level compiler integration.
//! - Model collections, checkpoint I/O, training, ONNX and serving utilities
//!   live in separately versioned crates rather than facade features.

pub use rxla_core as core;
pub use rxla_core::*;
pub use rxla_ir as ir;
pub use rxla_nn as nn;
pub use rxla_pjrt as pjrt;
