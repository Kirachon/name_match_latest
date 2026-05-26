<# 
.SYNOPSIS
    Compare two benchmark JSON files and enforce default regression gates.
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$Before,
  [Parameter(Mandatory = $true)]
  [string]$After,
  [string]$OutputMarkdown = "tmp/perf/results/comparison.md",
  [double]$MaxP95RegressionPercent = 5.0
)

$ErrorActionPreference = "Stop"

function Get-DatasetMap {
  param($Run)
  $map = @{}
  foreach ($dataset in $Run.datasets) {
    $map[$dataset.manifest.name] = $dataset
  }
  $map
}

$beforeRun = Get-Content -Raw -Path $Before | ConvertFrom-Json
$afterRun = Get-Content -Raw -Path $After | ConvertFrom-Json
$beforeMap = Get-DatasetMap $beforeRun
$afterMap = Get-DatasetMap $afterRun

$lines = @()
$lines += "# Benchmark Comparison"
$lines += ""
$lines += "| Dataset | Before p95 ms | After p95 ms | Delta % | Status |"
$lines += "|---|---:|---:|---:|---|"

$failed = $false
foreach ($name in ($beforeMap.Keys | Sort-Object)) {
  if (-not $afterMap.ContainsKey($name)) {
    $lines += "| $name | n/a | n/a | n/a | missing after run |"
    $failed = $true
    continue
  }

  $beforeP95 = [double]$beforeMap[$name].summary.warm_p95_elapsed_ms
  $afterP95 = [double]$afterMap[$name].summary.warm_p95_elapsed_ms
  if ($beforeP95 -le 0) {
    $delta = 0.0
  } else {
    $delta = (($afterP95 - $beforeP95) / $beforeP95) * 100.0
  }
  $status = if ($delta -gt $MaxP95RegressionPercent) { "fail" } else { "pass" }
  if ($status -eq "fail") { $failed = $true }
  $lines += "| $name | $beforeP95 | $afterP95 | {0:N2} | $status |" -f $delta
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputMarkdown) | Out-Null
$lines | Set-Content -Path $OutputMarkdown -Encoding UTF8
Write-Host "Wrote comparison to $OutputMarkdown"

if ($failed) {
  throw "Benchmark comparison failed regression gates."
}
