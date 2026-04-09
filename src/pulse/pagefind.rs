//! Pagefind binary management: download, cache, and version-check
//!
//! Downloads a pinned Pagefind release from GitHub and caches it at `~/.reflex/bin/pagefind`.
//! Handles platform detection, tar.gz extraction, and executable permissions.

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Pinned Pagefind version
const PAGEFIND_VERSION: &str = "1.5.0";

/// Return the path to the Pagefind binary, downloading it if needed.
///
/// 1. Check if `~/.reflex/bin/pagefind` exists
/// 2. If yes, verify version matches PAGEFIND_VERSION
/// 3. If no or version mismatch, download and extract
pub fn ensure_pagefind() -> Result<PathBuf> {
    let bin_dir = get_bin_dir()?;
    let pagefind_path = bin_dir.join("pagefind");

    if pagefind_path.exists() {
        // Check version
        if check_version(&pagefind_path)? {
            return Ok(pagefind_path);
        }
        eprintln!("Pagefind version mismatch, re-downloading...");
    }

    // Download and extract
    std::fs::create_dir_all(&bin_dir)
        .context("Failed to create ~/.reflex/bin/")?;

    download_pagefind(&pagefind_path)?;

    Ok(pagefind_path)
}

/// Get the ~/.reflex/bin/ directory path
fn get_bin_dir() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .context("Could not determine home directory")?;
    Ok(home.join(".reflex").join("bin"))
}

/// Check if the installed Pagefind binary matches the pinned version
fn check_version(pagefind_path: &PathBuf) -> Result<bool> {
    let output = std::process::Command::new(pagefind_path)
        .arg("--version")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            // pagefind --version outputs "pagefind 1.5.0" or just the version
            Ok(version_str.contains(PAGEFIND_VERSION))
        }
        _ => Ok(false),
    }
}

/// Determine the correct asset name for this platform
fn get_asset_name() -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let target = match (os, arch) {
        ("linux", "x86_64") => "x86_64-unknown-linux-musl",
        ("linux", "aarch64") => "aarch64-unknown-linux-musl",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        _ => anyhow::bail!(
            "Unsupported platform: {}-{}. Install Pagefind manually: https://pagefind.app/docs/installation/",
            os, arch
        ),
    };

    Ok(format!("pagefind-v{}-{}.tar.gz", PAGEFIND_VERSION, target))
}

/// Download Pagefind from GitHub releases and extract to the target path
fn download_pagefind(pagefind_path: &PathBuf) -> Result<()> {
    let asset_name = get_asset_name()?;
    let url = format!(
        "https://github.com/CloudCannon/pagefind/releases/download/v{}/{}",
        PAGEFIND_VERSION, asset_name
    );

    eprintln!("Downloading Pagefind v{} from {}...", PAGEFIND_VERSION, url);

    // Download using reqwest (blocking)
    let rt = tokio::runtime::Runtime::new()?;
    let bytes = rt.block_on(async {
        let response = reqwest::get(&url).await
            .context("Failed to download Pagefind")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download Pagefind: HTTP {} from {}",
                response.status(), url
            );
        }

        response.bytes().await
            .context("Failed to read Pagefind download")
    })?;

    eprintln!("Extracting Pagefind binary...");

    // Extract tar.gz
    let decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut archive = tar::Archive::new(decoder);

    let mut found = false;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;

        // Look for the pagefind binary in the archive
        if path.file_name().map(|n| n == "pagefind").unwrap_or(false) {
            let mut file = std::fs::File::create(pagefind_path)
                .context("Failed to create pagefind binary")?;
            std::io::copy(&mut entry, &mut file)?;
            found = true;
            break;
        }
    }

    if !found {
        anyhow::bail!("Could not find 'pagefind' binary in the downloaded archive");
    }

    // Set executable permission on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(pagefind_path, perms)
            .context("Failed to set executable permission on pagefind binary")?;
    }

    eprintln!("Pagefind v{} installed at {}", PAGEFIND_VERSION, pagefind_path.display());

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
        assert!(name.contains(PAGEFIND_VERSION));
        assert!(name.ends_with(".tar.gz"));
    }

    #[test]
    fn test_bin_dir() {
        let dir = get_bin_dir().unwrap();
        assert!(dir.ends_with(".reflex/bin"));
    }
}
