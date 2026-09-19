from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Iterator, TypeAlias

from google.protobuf.message import Message
from mcap.reader import make_reader
from mcap.writer import Writer

from . import events_pb2


_TOPICS: dict[type[Message], str] = {
    events_pb2.RunStarted: "/run/started",
    events_pb2.RunFinished: "/run/finished",
    events_pb2.TrainStep: "/train/step",
    events_pb2.Evaluation: "/eval/result",
    events_pb2.SystemMetrics: "/system/metrics",
    events_pb2.ProfileSpan: "/profile/span",
    events_pb2.CheckpointPublished: "/checkpoint/published",
    events_pb2.Diagnostic: "/diagnostic",
}
_NAMES = {message.DESCRIPTOR.full_name: message for message in _TOPICS}
DecodedEvent: TypeAlias = tuple[str, int, Message]


class RunLog:
    def __init__(self, path: str | Path, *, chunk_size: int = 1 << 20) -> None:
        self._stream: BinaryIO = Path(path).open("xb")
        self._writer = Writer(self._stream, chunk_size=chunk_size)
        self._writer.start(profile="rxla.log.v1", library="rxla-log-python/0.1.0a1")
        self._schema = files("rxla_log").joinpath("schema/events.desc").read_bytes()
        self._channels: dict[type[Message], int] = {}
        self._sequence = 0
        self._finished = False

    def write(self, event: Message, log_time_ns: int) -> None:
        event_type = type(event)
        try:
            topic = _TOPICS[event_type]
        except KeyError as error:
            raise TypeError(f"unsupported RXLA log event: {event_type!r}") from error
        channel_id = self._channels.get(event_type)
        if channel_id is None:
            schema_id = self._writer.register_schema(
                name=event.DESCRIPTOR.full_name,
                encoding="protobuf",
                data=self._schema,
            )
            channel_id = self._writer.register_channel(
                topic=topic,
                message_encoding="protobuf",
                schema_id=schema_id,
            )
            self._channels[event_type] = channel_id
        self._writer.add_message(
            channel_id=channel_id,
            log_time=log_time_ns,
            publish_time=log_time_ns,
            sequence=self._sequence,
            data=event.SerializeToString(),
        )
        self._sequence += 1

    def attach(
        self,
        name: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        log_time_ns: int = 0,
        create_time_ns: int | None = None,
    ) -> None:
        self._writer.add_attachment(
            create_time=log_time_ns if create_time_ns is None else create_time_ns,
            log_time=log_time_ns,
            name=name,
            media_type=media_type,
            data=data,
        )

    def finish(self) -> None:
        if self._finished:
            return
        self._writer.finish()
        self._stream.close()
        self._finished = True

    def flush(self) -> None:
        self._stream.flush()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish()


def iter_events(path: str | Path) -> Iterator[DecodedEvent]:
    with Path(path).open("rb") as stream:
        for schema, channel, message in make_reader(stream).iter_messages():
            if schema is None or schema.encoding != "protobuf":
                continue
            event_type = _NAMES.get(schema.name)
            if event_type is None:
                continue
            event = event_type()
            event.ParseFromString(message.data)
            yield channel.topic, message.log_time, event
