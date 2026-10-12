#[rxla_nn::model]
fn invalid() {
    #[layer()]
    let layer = ();
}

fn main() {}
