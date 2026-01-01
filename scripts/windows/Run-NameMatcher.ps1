param(
  [ValidateSet("remote", "local")]
  [string]$MySqlMode = "remote",

  [int]$RayonThreads = 12,
  [int]$GpuStreams = 2,

  [string]$ExePath = ".\\target\\release\\name_matcher.exe",

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

function Set-DefaultEnv([string]$Name, [string]$Value) {
  if (-not (Test-Path env:$Name)) {
    Set-Item -Path env:$Name -Value $Value
  }
}

# Safe performance defaults (override by pre-setting env vars).
Set-DefaultEnv "NAME_MATCHER_STREAMING" "1"
Set-DefaultEnv "NAME_MATCHER_AUTO_OPTIMIZE" "1"
Set-DefaultEnv "RAYON_NUM_THREADS" "$RayonThreads"

if ($MySqlMode -eq "local") {
  Set-DefaultEnv "NAME_MATCHER_POOL_SIZE" "24"
} else {
  Set-DefaultEnv "NAME_MATCHER_POOL_SIZE" "12"
}
Set-DefaultEnv "NAME_MATCHER_POOL_MIN" "4"

Set-DefaultEnv "NAME_MATCHER_USE_GPU" "1"
Set-DefaultEnv "NAME_MATCHER_GPU_STREAMS" "$GpuStreams"
Set-DefaultEnv "NAME_MATCHER_GPU_BUFFER_POOL" "1"
Set-DefaultEnv "NAME_MATCHER_GPU_PINNED_HOST" "1"

if (-not (Test-Path $ExePath)) {
  throw "Executable not found: $ExePath (build with: scripts\\windows\\Build-Release-Gpu.ps1)"
}

& $ExePath @Args

