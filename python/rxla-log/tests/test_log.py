from rxla_log import Metric, RunLog, TrainStep, iter_events
from pathlib import Path


def test_round_trip(tmp_path):
    path = tmp_path / "run.mcap"
    with RunLog(path) as log:
        log.write(
            TrainStep(
                step=7,
                loss=1.25,
                learning_rate=0.01,
                metrics=[Metric(name="box_loss", scalar=0.25)],
            ),
            123,
        )
        log.attach("sample.jpg", b"jpeg", media_type="image/jpeg", log_time_ns=124)

    events = list(iter_events(path))
    assert len(events) == 1
    topic, timestamp, event = events[0]
    assert topic == "/train/step"
    assert timestamp == 123
    assert event.step == 7
    assert event.metrics[0].name == "box_loss"


def test_reads_rust_fixture():
    path = Path(__file__).parent / "fixtures/rust.mcap"
    [(topic, timestamp, event)] = list(iter_events(path))
    assert topic == "/train/step"
    assert timestamp == 1_000
    assert event.step == 42
    assert event.metrics[0].name == "accuracy"
