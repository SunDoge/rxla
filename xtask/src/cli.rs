use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "xtask",
    about = "Maintainer checks and generated bindings for the RXLA workspace",
    version
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Clone, Copy, Subcommand)]
pub(crate) enum Command {
    /// Regenerate checked-in PJRT and protobuf Rust sources.
    Generate,
    /// Verify generated sources and the public release graph.
    Check,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_subcommands() {
        assert!(matches!(
            Cli::try_parse_from(["xtask", "generate"]).unwrap().command,
            Command::Generate
        ));
        assert!(matches!(
            Cli::try_parse_from(["xtask", "check"]).unwrap().command,
            Command::Check
        ));
    }

    #[test]
    fn rejects_unknown_or_extra_arguments() {
        assert!(Cli::try_parse_from(["xtask", "unknown"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "check", "extra"]).is_err());
    }
}
