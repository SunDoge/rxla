#[rxla_nn::model]
fn invalid() {
    #[layer("block.conv")]
    let layer = ();
}

fn main() {}
