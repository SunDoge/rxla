use std::{env, fs};

use rxla_log::{proto::TrainStep, read_flushed_events};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os()
        .nth(1)
        .ok_or("usage: latest_train_step LOG.mcap")?;
    let bytes = fs::read(path)?;
    let event = read_flushed_events::<TrainStep>(&bytes)?
        .pop()
        .ok_or("log contains no flushed training step")?;
    println!("{:#?}", event.event);
    Ok(())
}
