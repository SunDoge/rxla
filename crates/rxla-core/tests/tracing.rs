use rxla_core::{CacheLimits, Client, Compiler, Tracer};
use std::sync::{Arc, Mutex};
use tracing::{
    Subscriber,
    span::{Attributes, Id},
};
use tracing_subscriber::{Layer, layer::Context, prelude::*};

struct Spans(Arc<Mutex<Vec<&'static str>>>);
impl<S: Subscriber> Layer<S> for Spans {
    fn on_new_span(&self, attributes: &Attributes<'_>, _: &Id, _: Context<'_, S>) {
        self.0.lock().unwrap().push(attributes.metadata().name());
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_trace_distinguishes_cache_lookup_backend_and_transfers() {
    let spans = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(Spans(spans.clone()));
    tracing::subscriber::with_default(subscriber, || {
        let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
        let mut compiler = Compiler::new(client, CacheLimits::default());
        let graph = Tracer::default();
        let input = graph.input(&[2]).unwrap();
        let output = input.add_scalar(1.).unwrap();
        let executable = compiler.compile(&graph, &output).unwrap();
        compiler.compile(&graph, &output).unwrap();
        assert_eq!(executable.run(&[&[2., 3.]]).unwrap(), [3., 4.]);
    });
    let spans = spans.lock().unwrap();
    for (name, expected) in [
        ("xla.compile", 2),
        ("pjrt.compile", 1),
        ("pjrt.execute", 1),
        ("pjrt.host_to_device", 1),
        ("pjrt.device_to_host", 1),
    ] {
        assert_eq!(
            spans.iter().filter(|&&span| span == name).count(),
            expected,
            "{name}: {spans:?}"
        );
    }
}
