# Releasing RXLA

RXLA uses Conventional Commits. Create commits with Cocogitto when it is
available, for example:

```sh
cog commit feat "add an operation" core
cog commit fix "reject invalid layouts" pjrt
cog check
```

Plain `git commit` remains usable, but commit messages must have the form
`type(scope): description`. CI validates the complete history. Cocogitto owns
commit validation only; release-plz remains the sole owner of workspace version
updates, tags, GitHub releases, and crates.io publishing.

All public crates start at `0.1.0-alpha.1` and use exact internal dependency
requirements. Publish the first version manually because crates.io trusted
publishing cannot create a crate.

Publish in dependency order, waiting for crates.io to index each crate before
publishing its dependants:

```sh
cargo publish --registry crates-io -p rxla-cache
cargo publish --registry crates-io -p rxla-pjrt
cargo publish --registry crates-io -p rxla-xla-proto
cargo publish --registry crates-io -p rxla-ir
cargo publish --registry crates-io -p rxla-core
cargo publish --registry crates-io -p rxla-nn
cargo publish --registry crates-io -p rxla
```

`rxla-models`, `rxla-onnx`, `rxla-safetensors`, and `rxla-train` remain
workspace integration crates with `publish = false`. They continue to build and
test in CI but are intentionally outside the initial crates.io release graph.
Promoting one later requires removing that flag, adding it to `release-plz.toml`,
and reviewing its public dependency graph and packaged examples first.

After every crate exists, configure a crates.io trusted publisher for this
repository and the `.github/workflows/release-plz.yml` workflow. The workflow
requests an ephemeral OIDC token and intentionally has no
`CARGO_REGISTRY_TOKEN` secret.

Release-plz keeps RXLA packages in one version group, publishes changed crates
in dependency order, and creates one `v<version>` tag and prerelease for the
`rxla` facade. Changelogs remain manual during alpha development.
