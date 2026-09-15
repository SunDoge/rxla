use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn Error>>;
const BINDINGS: &str = "crates/rxla-pjrt/src/generated.rs";
const PROTOS: &str = "crates/rxla-xla-proto/src/generated";

pub(crate) fn run(check_only: bool) -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned();
    // Generate everything successfully before considering any source-tree write.
    // TempDir removes temporary artifacts on both success and ordinary failure.
    let temporary = tempfile::tempdir()?;
    generate(&root.join("vendor"), temporary.path())?;
    let expected = inventory(temporary.path())?;
    let actual = inventory(&root)?;
    let differences = differences(&expected, &actual);
    if check_only {
        if !differences.is_empty() {
            return Err(format!("generated files are out of date:\n{}\nRun xtask generate with the pinned toolchain and review the diff.", differences.join("\n")).into());
        }
        println!(
            "All {} generated files match; source tree unchanged.",
            expected.len()
        );
    } else {
        // Do not silently delete files when an upstream schema drops a package.
        let stale: Vec<_> = actual
            .keys()
            .filter(|path| !expected.contains_key(*path))
            .collect();
        if !stale.is_empty() {
            return Err(format!(
                "unexpected generated files; review/remove before regeneration: {stale:?}"
            )
            .into());
        }
        for (path, contents) in &expected {
            let destination = root.join(path);
            fs::create_dir_all(destination.parent().unwrap())?;
            fs::write(destination, contents)?;
        }
        println!(
            "Generated {} files; review changes before committing.",
            expected.len()
        );
    }
    Ok(())
}

fn generate(vendor: &Path, output: &Path) -> Result<()> {
    fs::create_dir_all(output.join("crates/rxla-pjrt/src"))?;
    fs::create_dir_all(output.join(PROTOS))?;
    let bindings = bindgen::Builder::default()
        .header(vendor.join("xla/pjrt/c/pjrt_c_api.h").to_string_lossy())
        .allowlist_type("PJRT_.*")
        .allowlist_var("PJRT_.*")
        .layout_tests(false)
        .generate_comments(false)
        .generate()?;
    bindings.write_to_file(output.join(BINDINGS))?;
    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    config.out_dir(output.join(PROTOS));
    let mut descriptors = config.load_fds(
        &[
            vendor.join("xla/service/hlo.proto"),
            vendor.join("xla/pjrt/proto/compile_options.proto"),
        ],
        &[vendor.to_owned(), protoc_bin_vendored::include_path()?],
    )?;
    // prost 0.14 does not support Editions yet. This upstream file contains
    // only a top-level enum, no message presence/default/encoding semantics.
    // Keep vendored sources intact and normalize only this checked descriptor.
    for file in &mut descriptors.file {
        if file.syntax.as_deref() == Some("editions") {
            if file.name.as_deref() != Some("xla/backends/autotuner/backends.proto")
                || !file.message_type.is_empty()
                || !file.extension.is_empty()
                || !file.service.is_empty()
            {
                return Err(
                    "unsupported Editions schema; review semantics before generating".into(),
                );
            }
            file.syntax = Some("proto3".into());
        }
    }
    config.compile_fds(descriptors)?;
    Ok(())
}

fn inventory(root: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>> {
    let mut files = BTreeMap::new();
    if root.join(BINDINGS).try_exists()? {
        files.insert(PathBuf::from(BINDINGS), fs::read(root.join(BINDINGS))?);
    }
    if root.join(PROTOS).try_exists()? {
        for entry in fs::read_dir(root.join(PROTOS))? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(format!(
                    "unexpected non-file in generated directory: {:?}",
                    entry.path()
                )
                .into());
            }
            let relative = Path::new(PROTOS).join(entry.file_name());
            files.insert(relative, fs::read(entry.path())?);
        }
    }
    Ok(files)
}

fn differences(
    expected: &BTreeMap<PathBuf, Vec<u8>>,
    actual: &BTreeMap<PathBuf, Vec<u8>>,
) -> Vec<String> {
    let mut differences = Vec::new();
    for (path, contents) in expected {
        match actual.get(path) {
            None => differences.push(format!("missing: {}", path.display())),
            Some(existing) if existing != contents => {
                differences.push(format!("changed: {}", path.display()))
            }
            _ => (),
        }
    }
    for path in actual.keys().filter(|path| !expected.contains_key(*path)) {
        differences.push(format!("unexpected: {}", path.display()));
    }
    differences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_changed_missing_and_stale_without_mutation() -> Result<()> {
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join(PROTOS))?;
        fs::write(root.path().join(PROTOS).join("xla.rs"), "old")?;
        fs::write(root.path().join(PROTOS).join("stale.rs"), "stale")?;
        let expected = BTreeMap::from([
            (PathBuf::from(BINDINGS), b"bindings".to_vec()),
            (Path::new(PROTOS).join("xla.rs"), b"new".to_vec()),
        ]);
        let before = inventory(root.path())?;
        let messages = differences(&expected, &before);
        assert_eq!(
            messages,
            [
                "missing: crates/rxla-pjrt/src/generated.rs",
                "changed: crates/rxla-xla-proto/src/generated/xla.rs",
                "unexpected: crates/rxla-xla-proto/src/generated/stale.rs",
            ]
        );
        assert_eq!(inventory(root.path())?, before);
        assert!(differences(&expected, &expected).is_empty());
        Ok(())
    }

    #[test]
    fn inventory_rejects_unexpected_directories() -> Result<()> {
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join(PROTOS).join("unexpected"))?;
        assert!(inventory(root.path()).is_err());
        Ok(())
    }
}
