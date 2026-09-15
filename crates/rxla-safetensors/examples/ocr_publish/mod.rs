//! Atomic visibility for Linux OCR example exports, not crash durability.
use std::path::Path;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(target_os = "linux")]
pub fn publish_new(target: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let name = target
        .file_name()
        .ok_or("artifact must name a new directory")?;
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let target = parent.join(name);
    match std::fs::symlink_metadata(&target) {
        Ok(_) => return Err("artifact destination already exists".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let staging = tempfile::Builder::new()
        .prefix(".ocr-artifact-")
        .tempdir_in(&parent)?;
    write(staging.path())?;
    // Unlike std::fs::rename, this cannot replace an existing empty directory
    // or a destination created by a concurrent exporter after preflight.
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        staging.path(),
        rustix::fs::CWD,
        &target,
        rustix::fs::RenameFlags::NOREPLACE,
    )?;
    // The staging name no longer exists. Do not attempt cleanup at its old path.
    let _ = staging.keep();
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn publish_new(_target: &Path, _write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    Err("atomic OCR artifact publication currently requires Linux renameat2".into())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn publishes_complete_package_and_cleans_failed_staging() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("model");
        assert!(
            publish_new(&target, |p| {
                fs::write(p.join("partial"), b"bytes")?;
                Err("injected writer failure".into())
            })
            .is_err()
        );
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        publish_new(&target, |p| {
            fs::write(p.join("executable.bin"), b"native")?;
            fs::write(p.join("manifest.json"), b"manifest")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(target.join("executable.bin")).unwrap(), b"native");
        assert_eq!(fs::read(target.join("manifest.json")).unwrap(), b"manifest");
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
        assert!(publish_new(&target, |_| panic!("must reject before writer")).is_err());
    }

    #[test]
    fn late_destinations_are_never_replaced() {
        for kind in 0..3 {
            let root = tempfile::tempdir().unwrap();
            let target = root.path().join("model");
            assert!(
                publish_new(&target, |p| {
                    fs::write(p.join("new"), b"new")?;
                    match kind {
                        0 => fs::create_dir(&target)?, // even an EMPTY directory must survive
                        1 => fs::write(&target, b"old")?,
                        _ => std::os::unix::fs::symlink("missing-target", &target)?,
                    }
                    Ok(())
                })
                .is_err()
            );
            match kind {
                0 => assert_eq!(fs::read_dir(&target).unwrap().count(), 0),
                1 => assert_eq!(fs::read(&target).unwrap(), b"old"),
                _ => assert_eq!(fs::read_link(&target).unwrap(), Path::new("missing-target")),
            }
            assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
        }
    }

    #[test]
    fn concurrent_publishers_have_exactly_one_winner() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("model");
        let barrier = std::sync::Barrier::new(2);
        let wins = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..2)
                .map(|i| {
                    let target = &target;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        publish_new(target, |p| {
                            fs::write(p.join("payload"), [i])?;
                            fs::write(p.join("manifest"), [i])?;
                            barrier.wait();
                            Ok(())
                        })
                        .is_ok()
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|job| usize::from(job.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(wins, 1);
        assert_eq!(
            fs::read(target.join("payload")).unwrap(),
            fs::read(target.join("manifest")).unwrap()
        );
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }
}
