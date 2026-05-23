param(
  [switch]$Clean,
  [switch]$Gpu,
  [switch]$UseGlobalCargoHome
)

$ErrorActionPreference = "Stop"

if (-not $UseGlobalCargoHome) {
  $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
  $env:CARGO_HOME = (Join-Path $repoRoot ".cargo-home")
}

if ($Clean) {
  cargo clean
}

if ($Gpu) {
  cargo build --release --locked --features gpu --bin gui
} else {
  cargo build --release --locked --bin gui
}
