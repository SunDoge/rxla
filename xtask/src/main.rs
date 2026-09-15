mod cli;
mod generator;

use clap::Parser;
use cli::{Cli, Command};

fn main() -> generator::Result<()> {
    let command = Cli::parse().command;
    generator::run(matches!(command, Command::Check))
}
