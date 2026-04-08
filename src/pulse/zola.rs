//! Zola binary management: download, cache, and version-check
//!
//! Downloads a pinned Zola release from GitHub and caches it at `~/.reflex/bin/zola`.
//! Handles platform detection, tar.gz extraction, and executable permissions.

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Pinned Zola version
const ZOLA_VERSION: &str = "0.19.2";

/// Return the path to the Zola binary, downloading it if needed.
///
/// 1. Check if `~/.reflex/bin/zola` exists
/// 2. If yes, verify version matches ZOLA_VERSION
/// 3. If no or version mismatch, download and extract
pub fn ensure_zola() -> Result<PathBuf> {
    let bin_dir = get_bin_dir()?;
    let zola_path = bin_dir.join("zola");

    if zola_path.exists() {
        // Check version
        if check_version(&zola_path)? {
            return Ok(zola_path);
        }
        eprintln!("Zola version mismatch, re-downloading...");
    }

    // Download and extract
    std::fs::create_dir_all(&bin_dir)
        .context("Failed to create ~/.reflex/bin/")?;

    download_zola(&zola_path)?;

    Ok(zola_path)
}

/// Get the ~/.reflex/bin/ directory path
fn get_bin_dir() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .context("Could not determine home directory")?;
    Ok(home.join(".reflex").join("bin"))
}

/// Check if the installed Zola binary matches the pinned version
fn check_version(zola_path: &PathBuf) -> Result<bool> {
    let output = std::process::Command::new(zola_path)
        .arg("--version")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            // zola --version outputs "zola 0.19.2"
            Ok(version_str.contains(ZOLA_VERSION))
        }
        _ => Ok(false),
    }
}

/// Determine the correct asset name for this platform
fn get_asset_name() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let target = match (os, arch) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        _ => anyhow::bail!(
            "Unsupported platform: {}-{}. Install Zola manually: https://www.getzola.org/documentation/getting-started/installation/",
            os, arch
        ),
    };

    Ok(format!("zola-v{}-{}.tar.gz", ZOLA_VERSION, target))
}

/// Download Zola from GitHub releases and extract to the target path
fn download_zola(zola_path: &PathBuf) -> Result<()> {
    let asset_name = get_asset_name()?;
    let url = format!(
        "https://github.com/getzola/zola/releases/download/v{}/{}",
        ZOLA_VERSION, asset_name
    );

    eprintln!("Downloading Zola v{} from {}...", ZOLA_VERSION, url);

    // Download using reqwest (blocking)
    let rt = tokio::runtime::Runtime::new()?;
    let bytes = rt.block_on(async {
        let response = reqwest::get(&url).await
            .context("Failed to download Zola")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download Zola: HTTP {} from {}",
                response.status(), url
            );
        }

        response.bytes().await
            .context("Failed to read Zola download")
    })?;

    eprintln!("Extracting Zola binary...");

    // Extract tar.gz
    let decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut archive = tar::Archive::new(decoder);

    let mut found = false;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;

        // Look for the zola binary in the archive
        if path.file_name().map(|n| n == "zola").unwrap_or(false) {
            let mut file = std::fs::File::create(zola_path)
                .context("Failed to create zola binary")?;
            std::io::copy(&mut entry, &mut file)?;
            found = true;
            break;
        }
    }

    if !found {
        anyhow::bail!("Could not find 'zola' binary in the downloaded archive");
    }

    // Set executable permission on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(zola_path, perms)
            .context("Failed to set executable permission on zola binary")?;
    }

    eprintln!("Zola v{} installed at {}", ZOLA_VERSION, zola_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_asset_name() {
        // Should succeed on supported platforms
        let result = get_asset_name();
        assert!(result.is_ok(), "Should detect platform: {:?}", result.err());
        let name = result.unwrap();
        assert!(name.contains(ZOLA_VERSION));
        assert!(name.ends_with(".tar.gz"));
    }

    #[test]
    fn test_bin_dir() {
        let dir = get_bin_dir().unwrap();
        assert!(dir.ends_with(".reflex/bin"));
    }
}
