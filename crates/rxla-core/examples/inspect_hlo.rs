//! Print StableHLO before compilation and a compact optimized-HLO inventory.
use rxla_core::{Client, Tracer};
use rxla_xla_proto::xla::HloModuleProto;

fn print_module(label: &str, module: &HloModuleProto) {
    println!("{label}: {}", module.name);
    for computation in &module.computations {
        println!(
            "  computation {} (root {})",
            computation.name, computation.root_id
        );
        for instruction in &computation.instructions {
            println!(
                "    {}: {} {:?} <- {:?}",
                instruction.id,
                instruction.opcode,
                instruction.shape.as_ref().map(|s| &s.dimensions),
                instruction.operand_ids
            );
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH")?)? };
    let graph = Tracer::default();
    let x = graph.input(&[4])?;
    let output = x.add_scalar(0.)?.mul_scalar(2.)?;
    println!("Frontend StableHLO:\n{}", graph.stablehlo(&output)?);
    let executable = graph.compile(&client, &output)?;
    print_module("Optimized HLO", &executable.optimized_hlo_proto()?);
    assert_eq!(executable.run(&[&[1., 2., 3., 4.]])?, [2., 4., 6., 8.]);
    Ok(())
}
