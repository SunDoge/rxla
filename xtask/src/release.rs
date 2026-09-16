use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::generator::Result;

const PUBLIC_PACKAGES: [&str; 7] = [
    "rxla-cache",
    "rxla-pjrt",
    "rxla-xla-proto",
    "rxla-ir",
    "rxla-core",
    "rxla-nn",
    "rxla",
];

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: BTreeSet<String>,
}

#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    version: String,
    publish: Option<Vec<String>>,
    dependencies: Vec<Dependency>,
}

impl Package {
    fn is_publishable(&self) -> bool {
        self.publish
            .as_ref()
            .is_none_or(|registries| !registries.is_empty())
    }
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    req: String,
    path: Option<PathBuf>,
    kind: Option<String>,
}

#[derive(Deserialize)]
struct ReleaseConfig {
    package: Vec<ReleasePackage>,
}

#[derive(Deserialize)]
struct ReleasePackage {
    name: String,
    version_group: Option<String>,
}

pub(crate) fn check() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is a workspace member")
        .to_owned();
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout)?;
    let release: ReleaseConfig =
        toml::from_str(&fs::read_to_string(root.join("release-plz.toml"))?)?;
    validate(&metadata, &release, &root)?;
    validate_package_files(&root)?;
    println!(
        "Public release packages, graph and publish order match: {}.",
        PUBLIC_PACKAGES.join(" -> ")
    );
    Ok(())
}

fn validate_package_files(root: &Path) -> Result<()> {
    for package in PUBLIC_PACKAGES {
        let output = Command::new(env!("CARGO"))
            .args(["package", "--locked", "--list", "-p", package])
            .current_dir(root)
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "could not assemble package {package:?}:\n{}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        if output.stdout.is_empty() {
            return Err(format!("package {package:?} contains no files").into());
        }
    }
    Ok(())
}

fn validate(metadata: &Metadata, release: &ReleaseConfig, root: &Path) -> Result<()> {
    let members = metadata
        .packages
        .iter()
        .filter(|package| metadata.workspace_members.contains(&package.id))
        .collect::<Vec<_>>();
    let actual = members
        .iter()
        .filter(|package| package.is_publishable())
        .map(|package| package.name.as_str())
        .collect::<BTreeSet<_>>();
    let expected = PUBLIC_PACKAGES.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "publishable workspace packages differ: expected {expected:?}, found {actual:?}"
        )
        .into());
    }

    let mut configured = BTreeSet::new();
    for package in &release.package {
        if !configured.insert(package.name.as_str()) {
            return Err(format!("duplicate release-plz package {:?}", package.name).into());
        }
        if package.version_group.as_deref() != Some("rxla") {
            return Err(format!(
                "release-plz package {:?} must use version_group = \"rxla\"",
                package.name
            )
            .into());
        }
    }
    if configured != expected {
        return Err(format!(
            "release-plz packages differ: expected {expected:?}, found {configured:?}"
        )
        .into());
    }
    let configured_order = release
        .package
        .iter()
        .map(|package| package.name.as_str())
        .collect::<Vec<_>>();
    if configured_order != PUBLIC_PACKAGES {
        return Err(format!(
            "release-plz packages must follow dependency order: expected {:?}, found {configured_order:?}",
            PUBLIC_PACKAGES
        )
        .into());
    }

    let by_name = members
        .iter()
        .map(|package| (package.name.as_str(), *package))
        .collect::<BTreeMap<_, _>>();
    let publish_position = PUBLIC_PACKAGES
        .into_iter()
        .enumerate()
        .map(|(position, name)| (name, position))
        .collect::<BTreeMap<_, _>>();
    for package in members
        .into_iter()
        .filter(|package| package.is_publishable())
    {
        for dependency in package.dependencies.iter().filter(|dependency| {
            dependency.path.is_some() && dependency.kind.as_deref() != Some("dev")
        }) {
            let Some(target) = by_name.get(dependency.name.as_str()) else {
                return Err(format!(
                    "public package {:?} has unknown workspace dependency {:?}",
                    package.name, dependency.name
                )
                .into());
            };
            if !target.is_publishable() {
                return Err(format!(
                    "public package {:?} depends on private package {:?}",
                    package.name, dependency.name
                )
                .into());
            }
            if publish_position[dependency.name.as_str()] >= publish_position[package.name.as_str()]
            {
                return Err(format!(
                    "public dependency {} -> {} is not in dependency-first publish order",
                    package.name, dependency.name
                )
                .into());
            }
            let required = format!("={}", target.version);
            if dependency.req != required {
                return Err(format!(
                    "public dependency {} -> {} must require {required:?}, found {:?}",
                    package.name, dependency.name, dependency.req
                )
                .into());
            }
            if !dependency
                .path
                .as_deref()
                .is_some_and(|path| root.join(path).try_exists().unwrap_or(false))
            {
                return Err(format!(
                    "public dependency {} -> {} has a missing local path",
                    package.name, dependency.name
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Metadata, ReleaseConfig) {
        let mut packages = PUBLIC_PACKAGES
            .into_iter()
            .map(|name| Package {
                id: name.to_owned(),
                name: name.to_owned(),
                version: "0.1.0-alpha.1".to_owned(),
                publish: None,
                dependencies: Vec::new(),
            })
            .collect::<Vec<_>>();
        packages.push(Package {
            id: "private".to_owned(),
            name: "private".to_owned(),
            version: "0.1.0-alpha.1".to_owned(),
            publish: Some(Vec::new()),
            dependencies: Vec::new(),
        });
        let workspace_members = packages.iter().map(|package| package.id.clone()).collect();
        let package = PUBLIC_PACKAGES
            .into_iter()
            .map(|name| ReleasePackage {
                name: name.to_owned(),
                version_group: Some("rxla".to_owned()),
            })
            .collect();
        (
            Metadata {
                packages,
                workspace_members,
            },
            ReleaseConfig { package },
        )
    }

    #[test]
    fn accepts_the_intended_release_graph() -> Result<()> {
        let (metadata, release) = fixture();
        validate(&metadata, &release, Path::new("."))
    }

    #[test]
    fn rejects_public_set_and_release_config_drift() {
        let (mut metadata, mut release) = fixture();
        metadata
            .packages
            .iter_mut()
            .find(|package| package.name == "private")
            .unwrap()
            .publish = None;
        assert!(validate(&metadata, &release, Path::new(".")).is_err());

        let (metadata, _) = fixture();
        release.package.pop();
        assert!(validate(&metadata, &release, Path::new(".")).is_err());
    }

    #[test]
    fn rejects_release_order_drift() {
        let (metadata, mut release) = fixture();
        release.package.swap(0, 3);
        assert!(validate(&metadata, &release, Path::new(".")).is_err());
    }

    #[test]
    fn rejects_non_exact_or_private_runtime_dependencies() {
        let root = tempfile::tempdir().unwrap();
        let (mut metadata, release) = fixture();
        let facade = metadata
            .packages
            .iter()
            .position(|package| package.name == "rxla")
            .unwrap();
        metadata.packages[facade].dependencies.push(Dependency {
            name: "rxla-core".to_owned(),
            req: "^0.1.0-alpha.1".to_owned(),
            path: Some(root.path().to_owned()),
            kind: None,
        });
        assert!(validate(&metadata, &release, root.path()).is_err());

        metadata.packages[facade].dependencies[0] = Dependency {
            name: "private".to_owned(),
            req: "=0.1.0-alpha.1".to_owned(),
            path: Some(root.path().to_owned()),
            kind: None,
        };
        assert!(validate(&metadata, &release, root.path()).is_err());
    }

    #[test]
    fn rejects_dependency_order_that_cannot_be_published() {
        let root = tempfile::tempdir().unwrap();
        let (mut metadata, release) = fixture();
        metadata
            .packages
            .iter_mut()
            .find(|package| package.name == "rxla-cache")
            .unwrap()
            .dependencies
            .push(Dependency {
                name: "rxla".to_owned(),
                req: "=0.1.0-alpha.1".to_owned(),
                path: Some(root.path().to_owned()),
                kind: None,
            });
        assert!(validate(&metadata, &release, root.path()).is_err());
    }
}
