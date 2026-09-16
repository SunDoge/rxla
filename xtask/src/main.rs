mod cli;
mod generator;
mod release;

use clap::Parser;
use cli::{Cli, Command};

fn main() -> generator::Result<()> {
    let command = Cli::parse().command;
    let check = matches!(command, Command::Check);
    generator::run(check)?;
    if check {
        release::check()?;
    }
    Ok(())
}
