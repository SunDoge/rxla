use rxla_core::{Client, Graph};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_optimized_hlo_is_owned_and_executable_survives_inspection() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let graph = Graph::default();
    let x = graph.input(&[4]).unwrap();
    // XLA should remove the addition by zero (the frontend does not).
    let y = x.add_scalar(0.).unwrap().mul_scalar(2.).unwrap();
    let input_stablehlo = graph.stablehlo(&y).unwrap();
    let exe = graph.compile(&client, &y).unwrap();
    let optimized = exe.optimized_hlo_proto().unwrap();
    assert!(!optimized.computations.is_empty());
    assert_eq!(
        optimized
            .host_program_shape
            .as_ref()
            .unwrap()
            .parameters
            .len(),
        1
    );
    assert!(input_stablehlo.contains("stablehlo.add"));
    assert!(
        !optimized
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .any(|i| i.opcode == "add")
    );
    let entry = optimized
        .computations
        .iter()
        .find(|c| c.id == optimized.entry_computation_id)
        .unwrap();
    assert!(entry.instructions.iter().any(|i| i.id == entry.root_id));
    let entry_ops: Vec<_> = entry
        .instructions
        .iter()
        .map(|i| i.opcode.as_str())
        .collect();
    eprintln!("Optimized entry operations: {entry_ops:?}");
    assert_eq!(exe.run(&[&[1., 2., 3., 4.]]).unwrap(), [2., 4., 6., 8.]);
    assert_eq!(exe.optimized_hlo_proto().unwrap(), optimized);
    drop(exe);
    drop(graph);
    drop(client);
    // All protobuf data remains owned after the source executable is gone.
    assert!(!optimized.name.is_empty());
    assert_eq!(
        optimized
            .host_program_shape
            .unwrap()
            .result
            .unwrap()
            .dimensions,
        [4]
    );
}
