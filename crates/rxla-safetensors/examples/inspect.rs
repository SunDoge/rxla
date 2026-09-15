use rxla_safetensors::SafeTensors;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: inspect <checkpoint.safetensors> [tensor-name]")?;
    let mut checkpoint = SafeTensors::open(path)?;
    println!("Tensors: {}", checkpoint.names().len());
    if let Some(name) = args.next() {
        println!("{name}: {:?}", checkpoint.info(&name));
        let tensor = checkpoint.read_f32(&name)?;
        println!(
            "F32 shape {:?}, first values {:?}",
            tensor.shape,
            &tensor.values[..tensor.values.len().min(8)]
        );
    }
    Ok(())
}
