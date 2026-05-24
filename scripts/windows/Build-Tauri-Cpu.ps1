#requires -Version 5.0
<#
.SYNOPSIS
    Build the Tauri v2 desktop app for Windows (CPU-only).

.DESCRIPTION
    Runs `pnpm install` (if node_modules is missing), `pnpm build`, then
    `cargo tauri build` against the path-dep `name_matcher` engine. The
    resulting artefact lands at:
      src-tauri\target\release\name-matcher-tauri.exe
    (or under src-tauri\target\release\bundle\ when --no-bundle is removed).

.PARAMETER Clean
    Remove `target/`, `dist/`, and `node_modules/` before building.

.PARAMETER UseGlobalCargoHome
    Use %USERPROFILE%\.cargo\ instead of -CargoHome.

.PARAMETER CargoHome
    Sandboxed CARGO_HOME to use (defaults to C:\cargo_nm_temp to avoid the
    locked in-tree .cargo-home cache).

.EXAMPLE
    powershell -File scripts\windows\Build-Tauri-Cpu.ps1
#>
param(
  [switch]$Clean,
  [switch]$UseGlobalCargoHome,
  [string]$CargoHome = "C:\cargo_nm_temp"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Write-Host "Repo root: $repoRoot"

if (-not $UseGlobalCargoHome) {
  if (-not (Test-Path $CargoHome)) {
    New-Item -ItemType Directory -Path $CargoHome | Out-Null
  }
  $env:CARGO_HOME = $CargoHome
  Write-Host "Using CARGO_HOME = $env:CARGO_HOME"
}

if ($Clean) {
  Write-Host "[clean] removing build artefacts"
  Remove-Item -Recurse -Force "$repoRoot\src-tauri\target" -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force "$repoRoot\ui\dist" -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force "$repoRoot\ui\node_modules" -ErrorAction SilentlyContinue
}

# 1) Install JS deps if needed.
if (-not (Test-Path "$repoRoot\ui\node_modules")) {
  Write-Host "[ui] pnpm install"
  Push-Location "$repoRoot\ui"
  pnpm install
  Pop-Location
}

# 2) Build the front-end.
Write-Host "[ui] pnpm build"
Push-Location "$repoRoot\ui"
pnpm build
Pop-Location

# 3) Verify the Tauri CLI is installed.
if (-not (Get-Command "cargo-tauri" -ErrorAction SilentlyContinue)) {
  Write-Host "[tauri] cargo-tauri not found - installing v2 ..."
  cargo install tauri-cli --version "^2" --locked
}

# 4) Build the Tauri shell (CPU). --no-bundle skips MSI/NSIS packaging until
#    real brand icons are generated via `cargo tauri icon`.
Write-Host "[tauri] cargo tauri build (CPU)"
Push-Location "$repoRoot\src-tauri"
cargo tauri build --no-bundle
Pop-Location

Write-Host ""
Write-Host "Done. Artefact:"
Write-Host "  $repoRoot\src-tauri\target\release\name-matcher-tauri.exe"
