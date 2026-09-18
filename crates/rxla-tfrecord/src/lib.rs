//! TFRecord integration for RXLA.
//!
//! [`tfrecord_lite`] owns the format itself: framing, masked CRC32C, codecs,
//! and synchronous byte I/O. This crate is the integration point for RXLA
//! use cases such as dataset shards, remote storage, and TensorBoard event
//! logs. Those higher-level policies are added here only when exercised by a
//! real consumer.

pub use tfrecord_lite as framing;
pub use tfrecord_lite::{
    BytesCodec, DecodeRecordError, DecodedReader, Decoder, EncodeRecordError, EncodedWriter,
    Encoder, Error, Reader, ReaderOptions, Result, Writer,
};
