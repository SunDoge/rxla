//! Structured training and inference event logs stored in MCAP.

pub mod proto {
    include!("generated/rxla.log.v1.rs");
}

use mcap::{records::MessageHeader, write::Writer};
use prost::Message;
use snafu::{ResultExt, Snafu};
use std::{
    collections::BTreeMap,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

const PROTO_SCHEMA: &[u8] = include_bytes!("../schema/events.desc");

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("open log {:?}: {source}", path))]
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    #[snafu(display("write MCAP log: {source}"))]
    Mcap { source: mcap::McapError },
    #[snafu(display("decode Protobuf event: {source}"))]
    Decode { source: prost::DecodeError },
    #[snafu(display("event sequence is exhausted"))]
    SequenceExhausted,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub trait Event: Message {
    const TOPIC: &'static str;
    const MESSAGE_NAME: &'static str;
}

macro_rules! event {
    ($type:ty, $topic:literal, $name:literal) => {
        impl Event for $type {
            const TOPIC: &'static str = $topic;
            const MESSAGE_NAME: &'static str = $name;
        }
    };
}

event!(proto::RunStarted, "/run/started", "rxla.log.v1.RunStarted");
event!(
    proto::RunFinished,
    "/run/finished",
    "rxla.log.v1.RunFinished"
);
event!(proto::TrainStep, "/train/step", "rxla.log.v1.TrainStep");
event!(proto::Evaluation, "/eval/result", "rxla.log.v1.Evaluation");
event!(
    proto::SystemMetrics,
    "/system/metrics",
    "rxla.log.v1.SystemMetrics"
);
event!(
    proto::ProfileSpan,
    "/profile/span",
    "rxla.log.v1.ProfileSpan"
);
event!(
    proto::CheckpointPublished,
    "/checkpoint/published",
    "rxla.log.v1.CheckpointPublished"
);
event!(proto::Diagnostic, "/diagnostic", "rxla.log.v1.Diagnostic");

pub struct RunLog {
    writer: Writer<BufWriter<File>>,
    channels: BTreeMap<&'static str, u16>,
    sequence: u32,
}

impl RunLog {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(path.as_ref())
            .context(OpenSnafu {
                path: path.as_ref(),
            })?;
        let writer = mcap::WriteOptions::new()
            .library(concat!("rxla-log/", env!("CARGO_PKG_VERSION")))
            .create(BufWriter::new(file))
            .context(McapSnafu)?;
        Ok(Self {
            writer,
            channels: BTreeMap::new(),
            sequence: 0,
        })
    }

    pub fn write<E: Event>(&mut self, event: &E, log_time_ns: u64) -> Result<()> {
        let next_sequence = self
            .sequence
            .checked_add(1)
            .ok_or(Error::SequenceExhausted)?;
        let channel_id = match self.channels.get(E::TOPIC) {
            Some(id) => *id,
            None => {
                let schema_id = self
                    .writer
                    .add_schema(E::MESSAGE_NAME, "protobuf", PROTO_SCHEMA)
                    .context(McapSnafu)?;
                let channel_id = self
                    .writer
                    .add_channel(schema_id, E::TOPIC, "protobuf", &BTreeMap::new())
                    .context(McapSnafu)?;
                self.channels.insert(E::TOPIC, channel_id);
                channel_id
            }
        };
        let payload = event.encode_to_vec();
        self.writer
            .write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence: self.sequence,
                    log_time: log_time_ns,
                    publish_time: log_time_ns,
                },
                &payload,
            )
            .context(McapSnafu)?;
        self.sequence = next_sequence;
        Ok(())
    }

    pub fn attach(
        &mut self,
        name: impl Into<String>,
        media_type: impl Into<String>,
        data: &[u8],
        log_time_ns: u64,
    ) -> Result<()> {
        self.writer
            .attach(&mcap::Attachment {
                log_time: log_time_ns,
                create_time: log_time_ns,
                name: name.into(),
                media_type: media_type.into(),
                data: data.into(),
            })
            .context(McapSnafu)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context(McapSnafu)
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer.finish().context(McapSnafu)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimedEvent<E> {
    pub log_time_ns: u64,
    pub publish_time_ns: u64,
    pub sequence: u32,
    pub event: E,
}

/// Decode all events of one type from a mapped or otherwise resident MCAP.
pub fn read_events<E: Event + Default>(bytes: &[u8]) -> Result<Vec<TimedEvent<E>>> {
    read_events_from_stream(mcap::MessageStream::new(bytes).context(McapSnafu)?)
}

/// Decode events from a log that is still being written or lost its final MCAP footer.
///
/// Only complete, CRC-valid chunks are returned. A partial final record remains an error;
/// callers should retry after the writer's next [`RunLog::flush`] rather than treating
/// partially persisted data as an event.
pub fn read_flushed_events<E: Event + Default>(bytes: &[u8]) -> Result<Vec<TimedEvent<E>>> {
    use enumset::enum_set;
    use mcap::read::Options;

    read_events_from_stream(
        mcap::MessageStream::new_with_options(bytes, enum_set!(Options::IgnoreEndMagic))
            .context(McapSnafu)?,
    )
}

fn read_events_from_stream<E: Event + Default>(
    messages: mcap::MessageStream<'_>,
) -> Result<Vec<TimedEvent<E>>> {
    let mut events = Vec::new();
    for message in messages {
        let message = message.context(McapSnafu)?;
        if message.channel.topic != E::TOPIC {
            continue;
        }
        events.push(TimedEvent {
            log_time_ns: message.log_time,
            publish_time_ns: message.publish_time,
            sequence: message.sequence,
            event: E::decode(message.data.as_ref()).context(DecodeSnafu)?,
        });
    }
    Ok(events)
}

impl proto::Metric {
    pub fn scalar(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value: Some(proto::metric::Value::Scalar(value)),
            unit: None,
        }
    }

    pub fn integer(name: impl Into<String>, value: i64) -> Self {
        Self {
            name: name.into(),
            value: Some(proto::metric::Value::Integer(value)),
            unit: None,
        }
    }

    pub fn boolean(name: impl Into<String>, value: bool) -> Self {
        Self {
            name: name.into(),
            value: Some(proto::metric::Value::Boolean(value)),
            unit: None,
        }
    }

    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: Some(proto::metric::Value::Text(value.into())),
            unit: None,
        }
    }

    #[must_use]
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

impl proto::TrainStep {
    pub fn new(step: u64, loss: f64) -> Self {
        Self {
            step,
            loss,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    #[must_use]
    pub fn metric(mut self, metric: proto::Metric) -> Self {
        self.metrics.push(metric);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_indexed_protobuf_and_attachment() -> Result<()> {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("run.mcap");
        let mut log = RunLog::create(&path)?;
        log.write(
            &proto::TrainStep::new(7, 1.25)
                .learning_rate(0.01)
                .metric(proto::Metric::scalar("box_loss", 0.25)),
            123,
        )?;
        log.attach("sample.jpg", "image/jpeg", b"jpeg", 124)?;
        log.finish()?;

        let bytes = std::fs::read(path).expect("read MCAP");
        let events = read_events::<proto::TrainStep>(&bytes)?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.step, 7);
        assert_eq!(events[0].event.metrics[0].name, "box_loss");
        Ok(())
    }

    #[test]
    fn reads_complete_chunks_before_log_is_finished() -> Result<()> {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("live.mcap");
        let mut log = RunLog::create(&path)?;
        log.write(&proto::TrainStep::new(100, 2.5), 123)?;
        log.flush()?;

        let bytes = std::fs::read(&path).expect("read live MCAP");
        let events = read_flushed_events::<proto::TrainStep>(&bytes)?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.step, 100);

        // Keep the writer alive until after the incomplete-file assertion.
        log.finish()?;
        Ok(())
    }

    #[test]
    fn reads_python_fixture() {
        let bytes = include_bytes!("../../../python/rxla-log/tests/fixtures/python.mcap");
        let message = mcap::MessageStream::new(bytes)
            .expect("valid Python MCAP")
            .next()
            .expect("one event")
            .expect("valid event");
        assert_eq!(message.channel.topic, proto::TrainStep::TOPIC);
        let event = proto::TrainStep::decode(message.data.as_ref()).expect("decode Python event");
        assert_eq!(event.step, 7);
        assert_eq!(event.metrics[0].name, "box_loss");
    }
}
