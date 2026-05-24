#requires -Version 5.0
<#
.SYNOPSIS
    Build the legacy egui GUI binary (release).

.DESCRIPTION
    Compiles the legacy `gui` binary which is now gated behind the `gui`
    Cargo feature. With `-Gpu`, the `gpu` feature is also enabled so the
    binary links against `cudarc` and uses CUDA where available.

.PARAMETER Clean
    `cargo clean` before building.

.PARAMETER Gpu
    Pass `--features "gui,gpu"` instead of `--features "gui"`.

.PARAMETER UseGlobalCargoHome
    Use `%USERPROFILE%\.cargo` instead of -CargoHome.

.PARAMETER CargoHome
    Sandboxed CARGO_HOME (defaults to C:\cargo_nm_temp to avoid the locked
    in-tree .cargo-home cache).

.EXAMPLE
    powershell -File scripts\windows\Build-Release-Gui.ps1
    powershell -File scripts\windows\Build-Release-Gui.ps1 -Gpu
#>
param(
  [switch]$Clean,
  [switch]$Gpu,
  [switch]$UseGlobalCargoHome,
  [string]$CargoHome = "C:\cargo_nm_temp"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

if (-not $UseGlobalCargoHome) {
  if (-not (Test-Path $CargoHome)) {
    New-Item -ItemType Directory -Path $CargoHome | Out-Null
  }
  $env:CARGO_HOME = $CargoHome
  Write-Host "Using CARGO_HOME = $env:CARGO_HOME"
}

if ($Clean) {
  cargo clean
}

# Legacy egui GUI now requires the `gui` feature flag so the default
# build does not pull eframe/egui/rfd. -Gpu adds `gpu` for CUDA.
if ($Gpu) {
  $features = "gui,gpu"
  # Verify nvcc — fail fast for clearer error than cudarc panics.
  $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
  if (-not $nvcc) {
    Write-Error "nvcc not found on PATH. Install CUDA Toolkit 12.x and retry."
    exit 2
  }
  Write-Host "nvcc: $($nvcc.Source)"
} else {
  $features = "gui"
}

Push-Location $repoRoot
cargo build --release --locked --features $features --bin gui
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { exit $rc }

$exe = "$repoRoot\target\release\gui.exe"
if (Test-Path $exe) {
  $info = Get-Item $exe
  Write-Host ("Built: {0} ({1:N0} bytes)" -f $exe, $info.Length)
}
