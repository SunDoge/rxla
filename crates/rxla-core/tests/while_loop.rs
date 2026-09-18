use rxla_core::{Client, Tensor, Tracer};

fn loop_program(graph: &Tracer) -> Vec<Tensor> {
    let counter = graph.input_i32_scalar().unwrap();
    let values = graph.input(&[2]).unwrap();
    Tensor::while_loop(
        &[counter, values],
        |carry| Ok(carry[0].clone()),
        |carry| Ok(vec![carry[0].scalar_i32(0)?, carry[1].add(&carry[1])?]),
    )
    .unwrap()
}

#[test]
fn while_loop_is_one_structured_ir_region() {
    let graph = Tracer::default();
    let outputs = loop_program(&graph);
    let stablehlo = graph.stablehlo_many(&outputs).unwrap();
    assert_eq!(stablehlo.matches("stablehlo.while").count(), 1);
    assert_eq!(stablehlo.matches("stablehlo.add").count(), 1);
    assert!(stablehlo.contains("cond {"));
    assert!(stablehlo.contains("} do {"));
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn while_loop_executes_with_multiple_loop_carries() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let outputs = loop_program(&graph);
    let executable = graph.compile_many(&client, &outputs).unwrap();
    let counter = client.buffer(&[], &[1_i32]).unwrap();
    let values = client.buffer(&[2], &[3_f32, 5.]).unwrap();
    let actual = executable.execute(&[&counter, &values]).unwrap();
    assert_eq!(actual[0].to_vec::<i32>().unwrap(), [0]);
    assert_eq!(actual[1].to_vec::<f32>().unwrap(), [6., 10.]);
}
