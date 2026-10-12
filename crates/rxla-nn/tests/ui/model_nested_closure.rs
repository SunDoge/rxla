#[rxla_nn::model]
fn invalid() {
    let nested = || {
        #[layer("conv")]
        let layer = ();
    };
}

fn main() {}
