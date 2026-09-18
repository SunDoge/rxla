use rxla_core::{Client, DType, Tracer};

#[test]
fn searchsorted_builds_logarithmic_static_graph() {
    let graph = Tracer::default();
    let values = graph.input_i32([7]).unwrap();
    let needles = graph.input_i32([2, 3]).unwrap();
    let left = values.searchsorted_i32(&needles, false).unwrap();
    let right = values.searchsorted_i32(&needles, true).unwrap();
    assert_eq!(left.shape(), [2, 3]);
    assert_eq!(right.shape(), [2, 3]);
    let stablehlo = graph.stablehlo_many(&[left, right]).unwrap();
    assert!(stablehlo.contains("stablehlo.gather"));
    assert!(stablehlo.contains("stablehlo.select"));
}

#[test]
fn f32_to_i32_cast_is_nondifferentiable() {
    let graph = Tracer::default();
    let input = graph.input([3]).unwrap();
    let integers = input.floor().unwrap().cast(DType::I32).unwrap();
    assert_eq!(integers.dtype(), DType::I32);
    assert!(integers.grad(std::slice::from_ref(&input)).is_err());
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_searchsorted_handles_duplicates_edges_and_matrix_needles() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Tracer::default();
    let values = graph.input_i32([7]).unwrap();
    let needles = graph.input_i32([2, 3]).unwrap();
    let outputs = [
        values.searchsorted_i32(&needles, false).unwrap(),
        values.searchsorted_i32(&needles, true).unwrap(),
    ];
    let executable = graph.compile_many(&client, &outputs).unwrap();
    let values = client.buffer(&[7], &[-3, 0, 0, 2, 5, 5, 9]).unwrap();
    let needles = client.buffer(&[2, 3], &[-4, 0, 1, 5, 9, 10]).unwrap();
    let outputs = executable.execute(&[&values, &needles]).unwrap();
    assert_eq!(outputs[0].to_vec::<i32>().unwrap(), [0, 1, 3, 4, 6, 7]);
    assert_eq!(outputs[1].to_vec::<i32>().unwrap(), [0, 3, 3, 6, 7, 7]);
}
