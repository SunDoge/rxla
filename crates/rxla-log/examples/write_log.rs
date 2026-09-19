use rxla_log::{RunLog, proto};

fn main() -> rxla_log::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .expect("usage: write_log OUTPUT.mcap");
    let mut log = RunLog::create(path)?;
    log.write(
        &proto::TrainStep::new(42, 0.75)
            .learning_rate(0.001)
            .metric(proto::Metric::scalar("accuracy", 0.9)),
        1_000,
    )?;
    log.attach("preview.webp", "image/webp", b"webp", 1_001)?;
    log.finish()
}
