//! Subscriber policy belongs to the application, never the tensor library.
use rxla_core::{Client, Runtime, Tracer};
use tracing_subscriber::fmt::format::FmtSpan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_span_events(FmtSpan::CLOSE)
        .with_ansi(false)
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    let mut runtime = Runtime::new(client)?;
    let program = Tracer::trace(|trace| {
        let input = trace.input(&[2])?;
        Ok(vec![input.add_scalar(1.)?])
    })?;
    let executable = program.compile(&mut runtime)?;
    program.compile(&mut runtime)?;
    assert_eq!(executable.run(&[&[2., 3.]])?, [3., 4.]);
    Ok(())
}
