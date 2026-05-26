<#
.SYNOPSIS
    Run repeatable name matcher performance baselines and emit JSON evidence.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\perf\Run-Benchmarks.ps1 `
      -CommandTemplate 'target\release\name_matcher --source "{source}" --target "{target}"' `
      -BuildLabel cpu-release
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$CommandTemplate,
  [string]$DatasetDir = "tmp/perf/datasets",
  [string]$OutputPath = "tmp/perf/results/baseline.json",
  [string]$BuildLabel = "unspecified",
  [int]$RepeatCount = 5,
  [string[]]$DatasetNames = @(),
  [int]$CommandTimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"

function Get-GitValue {
  param([string[]]$GitArgs)
  try {
    (& git @GitArgs 2>$null) -join "`n"
  } catch {
    $null
  }
}

function Invoke-BenchmarkCommand {
  param(
    [string]$Command,
    [string]$WorkingDirectory
  )

  $wrappedCommand = "`$ProgressPreference = 'SilentlyContinue'; $Command"
  $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($wrappedCommand))
  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = "powershell"
  $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedCommand"
  $psi.WorkingDirectory = $WorkingDirectory
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true

  $process = [System.Diagnostics.Process]::new()
  $process.StartInfo = $psi
  $watch = [System.Diagnostics.Stopwatch]::StartNew()
  [void]$process.Start()
  $outTask = $process.StandardOutput.ReadToEndAsync()
  $errTask = $process.StandardError.ReadToEndAsync()
  if (-not $process.WaitForExit($CommandTimeoutSeconds * 1000)) {
    try { $process.Kill($true) } catch { $process.Kill() }
    $watch.Stop()
    return [ordered]@{
      command = $Command
      exit_code = -1
      elapsed_ms = [int64]$watch.ElapsedMilliseconds
      stdout_excerpt = ""
      stderr_excerpt = "Command timed out after $CommandTimeoutSeconds seconds."
    }
  }
  $watch.Stop()

  $outText = $outTask.GetAwaiter().GetResult()
  $errText = $errTask.GetAwaiter().GetResult()

  [ordered]@{
    command = $Command
    exit_code = $process.ExitCode
    elapsed_ms = [int64]$watch.ElapsedMilliseconds
    stdout_excerpt = if ($outText.Length -gt 4000) { $outText.Substring(0, 4000) } else { $outText }
    stderr_excerpt = if ($errText.Length -gt 4000) { $errText.Substring(0, 4000) } else { $errText }
  }
}

$repoRoot = (Resolve-Path ".").Path
$datasetRoot = Resolve-Path $DatasetDir
$manifests = Get-ChildItem -Path $datasetRoot -Recurse -Filter manifest.json
if ($DatasetNames.Count -gt 0) {
  $nameSet = @{}
  foreach ($name in $DatasetNames) { $nameSet[$name] = $true }
  $manifests = $manifests | Where-Object {
    $manifest = Get-Content -Raw -Path $_.FullName | ConvertFrom-Json
    $nameSet.ContainsKey($manifest.name)
  }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null

$run = [ordered]@{
  schema_version = 1
  generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  repo_root = $repoRoot
  git_commit = Get-GitValue -GitArgs @("rev-parse", "HEAD")
  git_branch = Get-GitValue -GitArgs @("branch", "--show-current")
  git_status_short = Get-GitValue -GitArgs @("status", "--short")
  build_label = $BuildLabel
  repeat_count = $RepeatCount
  cold_runs_per_dataset = 1
  environment = [ordered]@{
    os = [System.Environment]::OSVersion.VersionString
    machine = $env:COMPUTERNAME
    processor_count = [System.Environment]::ProcessorCount
    powershell = $PSVersionTable.PSVersion.ToString()
    cuda_path = $env:CUDA_PATH
    name_matcher_perf_log = $env:NAME_MATCHER_PERF_LOG
  }
  datasets = @()
}

foreach ($manifestPath in $manifests) {
  $manifest = Get-Content -Raw -Path $manifestPath.FullName | ConvertFrom-Json
  $outDir = Join-Path (Split-Path -Parent $OutputPath) $manifest.name
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  $commandBase = $CommandTemplate.
    Replace("{source}", $manifest.source).
    Replace("{target}", $manifest.target).
    Replace("{dataset}", $manifest.name).
    Replace("{output}", $outDir)

  $datasetResult = [ordered]@{
    manifest = $manifest
    cold = Invoke-BenchmarkCommand -Command $commandBase -WorkingDirectory $repoRoot
    warm = @()
  }

  for ($i = 1; $i -le $RepeatCount; $i++) {
    $datasetResult.warm += Invoke-BenchmarkCommand -Command $commandBase -WorkingDirectory $repoRoot
  }

  $elapsed = @($datasetResult.warm | ForEach-Object { $_.elapsed_ms } | Sort-Object)
  if ($elapsed.Count -gt 0) {
    $p50Index = [math]::Min($elapsed.Count - 1, [math]::Floor(($elapsed.Count - 1) * 0.50))
    $p95Index = [math]::Min($elapsed.Count - 1, [math]::Ceiling(($elapsed.Count - 1) * 0.95))
    $p99Index = [math]::Min($elapsed.Count - 1, [math]::Ceiling(($elapsed.Count - 1) * 0.99))
    $datasetResult.summary = [ordered]@{
      warm_p50_elapsed_ms = $elapsed[$p50Index]
      warm_p95_elapsed_ms = $elapsed[$p95Index]
      warm_p99_elapsed_ms = $elapsed[$p99Index]
      warm_min_elapsed_ms = $elapsed[0]
      warm_max_elapsed_ms = $elapsed[$elapsed.Count - 1]
    }
  }

  $run.datasets += $datasetResult
}

$run | ConvertTo-Json -Depth 20 | Set-Content -Path $OutputPath -Encoding UTF8
Write-Host "Wrote benchmark evidence to $OutputPath"
