//! CPU-side PP-OCRv6 pipeline utilities.
//!
//! Neural-network lowering lives separately from these explicit host operations.

mod preprocess;
pub use preprocess::{DetectorImage, RecognitionBatch, RecognitionPreprocessor};

mod ctc;
pub use ctc::{CtcDecoder, DecodedSequence};

mod error;
pub use error::{Error, Result};
