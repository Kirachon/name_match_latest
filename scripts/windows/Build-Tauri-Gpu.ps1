#requires -Version 5.0
<#
.SYNOPSIS
    Build the Tauri v2 desktop app for Windows with the GPU feature enabled.

.DESCRIPTION
    Same flow as `Build-Tauri-Cpu.ps1` but forwards `--features gpu` to the
    `name_matcher` path-dep so the engine compiles CUDA kernels. Requires
    nvcc (CUDA Toolkit 12.x) on PATH and a CUDA-capable Windows host.

.EXAMPLE
    powershell -File scripts\windows\Build-Tauri-Gpu.ps1
#>
param(
  [switch]$Clean,
  [switch]$UseGlobalCargoHome,
  [switch]$NoBundle,
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
}

# Verify nvcc - fail fast for clearer error than cudarc panics.
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
  Write-Error "nvcc not found on PATH. Install CUDA Toolkit 12.x and retry."
  exit 2
}
Write-Host "nvcc: $($nvcc.Source)"
nvcc --version | Select-Object -First 4

if ($Clean) {
  Remove-Item -Recurse -Force "$repoRoot\src-tauri\target" -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force "$repoRoot\ui\dist" -ErrorAction SilentlyContinue
}

if (-not (Test-Path "$repoRoot\ui\node_modules")) {
  Push-Location "$repoRoot\ui"; pnpm install; Pop-Location
}
Push-Location "$repoRoot\ui"; pnpm build; Pop-Location

if (-not (Get-Command cargo-tauri -ErrorAction SilentlyContinue)) {
  cargo install tauri-cli --version "^2" --locked
}

# GPU runtime DLLs - discover exact CUDA 12.x filenames from the prepared
# runtime folder so minor CUDA Toolkit changes do not break packaging.
$dist = "$repoRoot\src-tauri\target\release"
New-Item -ItemType Directory -Force $dist | Out-Null
$expected = @("nvrtc64_*.dll", "nvrtc-builtins64_*.dll")
$copied = @()
foreach ($pattern in $expected) {
  $matches = Get-ChildItem "$repoRoot\dist\gpu-dlls" -Filter $pattern -File -ErrorAction SilentlyContinue
  if ($matches.Count -gt 0) {
    foreach ($m in $matches) {
      Copy-Item $m.FullName $dist -Force
      $copied += $m.Name
    }
  } else {
    Write-Error "Missing required GPU runtime DLL matching: $pattern"
    exit 1
  }
}
Write-Host "Copied GPU DLLs: $($copied -join ', ')"

Write-Host "[tauri] cargo tauri build (GPU)"
Push-Location "$repoRoot\src-tauri"
if ($NoBundle) {
  cargo tauri build --features gpu --no-bundle
} else {
  cargo tauri build --features gpu
}
Pop-Location

# Ensure portable release-folder artifacts include the runtime DLLs even when
# the bundler rebuilds the EXE after the first copy.
foreach ($pattern in $expected) {
  Get-ChildItem "$repoRoot\dist\gpu-dlls" -Filter $pattern -File -ErrorAction SilentlyContinue |
    ForEach-Object { Copy-Item $_.FullName $dist -Force }
}

Write-Host ""
Write-Host "Done. Verify:"
Write-Host "  Get-ChildItem '$dist' -Filter *.dll"
