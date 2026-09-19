# rxla-log

Python reader and writer for RXLA's Protobuf events in indexed MCAP logs.

The wire schema is shared with the Rust `rxla-log` crate. Large binary values
such as images, compiler IR, and traces should be MCAP attachments rather than
embedded metrics.
