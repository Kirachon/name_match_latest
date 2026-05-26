<# 
.SYNOPSIS
    Compare two match-output JSON files for recall/parity regressions.

.DESCRIPTION
    Inputs can be either a JSON array of rows or an object with a `rows` array.
    Each row is expected to include source/target ids, confidence, matched level,
    label/method, and matched fields when available.
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$Before,
  [Parameter(Mandatory = $true)]
  [string]$After,
  [string]$OutputMarkdown = "tmp/perf/results/recall-comparison.md",
  [double]$ConfidenceTolerance = 0.001,
  [int]$AllowedFalseNegatives = 0,
  [switch]$RequireStableOrder
)

$ErrorActionPreference = "Stop"

function Get-Rows {
  param([string]$Path)
  $json = Get-Content -Raw -Path $Path | ConvertFrom-Json
  if ($json -is [array]) { return @($json) }
  if ($null -ne $json.rows) { return @($json.rows) }
  throw "JSON input $Path must be an array or contain a rows array."
}

function Get-Field {
  param($Row, [string[]]$Names)
  foreach ($name in $Names) {
    if ($Row.PSObject.Properties.Name -contains $name) {
      return $Row.$name
    }
  }
  return $null
}

function Get-Key {
  param($Row)
  $source = Get-Field $Row @("source_id", "sourceId", "source")
  $target = Get-Field $Row @("target_id", "targetId", "target")
  if ($null -eq $source -or $null -eq $target) {
    throw "Row is missing source_id/sourceId or target_id/targetId."
  }
  "$source|$target"
}

function Normalize-Fields {
  param($Fields)
  if ($null -eq $Fields) { return "" }
  [string]::Join(",", (@($Fields) | Sort-Object | ForEach-Object { "$_" }))
}

$beforeRows = Get-Rows $Before
$afterRows = Get-Rows $After
$beforeByKey = @{}
$afterByKey = @{}

for ($i = 0; $i -lt $beforeRows.Count; $i++) {
  $beforeByKey[(Get-Key $beforeRows[$i])] = [pscustomobject]@{ Row = $beforeRows[$i]; Index = $i }
}
for ($i = 0; $i -lt $afterRows.Count; $i++) {
  $afterByKey[(Get-Key $afterRows[$i])] = [pscustomobject]@{ Row = $afterRows[$i]; Index = $i }
}

$missing = New-Object System.Collections.Generic.List[string]
$unexpected = New-Object System.Collections.Generic.List[string]
$changed = New-Object System.Collections.Generic.List[string]

foreach ($key in $beforeByKey.Keys) {
  if (-not $afterByKey.ContainsKey($key)) {
    $missing.Add($key)
    continue
  }
  $beforeRow = $beforeByKey[$key].Row
  $afterRow = $afterByKey[$key].Row
  $beforeConfidence = [double](Get-Field $beforeRow @("confidence", "score"))
  $afterConfidence = [double](Get-Field $afterRow @("confidence", "score"))
  $beforeLevel = Get-Field $beforeRow @("matched_at_level", "matchedAtLevel", "level")
  $afterLevel = Get-Field $afterRow @("matched_at_level", "matchedAtLevel", "level")
  $beforeFields = Normalize-Fields (Get-Field $beforeRow @("matched_fields", "matchedFields"))
  $afterFields = Normalize-Fields (Get-Field $afterRow @("matched_fields", "matchedFields"))

  if ([math]::Abs($afterConfidence - $beforeConfidence) -gt $ConfidenceTolerance) {
    $changed.Add("$key confidence $beforeConfidence -> $afterConfidence")
  }
  if ("$beforeLevel" -ne "$afterLevel") {
    $changed.Add("$key level $beforeLevel -> $afterLevel")
  }
  if ($beforeFields -ne $afterFields) {
    $changed.Add("$key matched_fields changed")
  }
  if ($RequireStableOrder -and $beforeByKey[$key].Index -ne $afterByKey[$key].Index) {
    $changed.Add("$key order $($beforeByKey[$key].Index) -> $($afterByKey[$key].Index)")
  }
}

foreach ($key in $afterByKey.Keys) {
  if (-not $beforeByKey.ContainsKey($key)) {
    $unexpected.Add($key)
  }
}

$lines = @()
$lines += "# Recall Comparison"
$lines += ""
$lines += "| Metric | Count |"
$lines += "|---|---:|"
$lines += "| before_rows | $($beforeRows.Count) |"
$lines += "| after_rows | $($afterRows.Count) |"
$lines += "| missing_pairs | $($missing.Count) |"
$lines += "| unexpected_pairs | $($unexpected.Count) |"
$lines += "| changed_pairs | $($changed.Count) |"
$lines += ""

if ($missing.Count -gt 0) {
  $lines += "## Missing Pairs"
  $missing | Select-Object -First 50 | ForEach-Object { $lines += "- $_" }
  $lines += ""
}
if ($unexpected.Count -gt 0) {
  $lines += "## Unexpected Pairs"
  $unexpected | Select-Object -First 50 | ForEach-Object { $lines += "- $_" }
  $lines += ""
}
if ($changed.Count -gt 0) {
  $lines += "## Changed Pairs"
  $changed | Select-Object -First 50 | ForEach-Object { $lines += "- $_" }
  $lines += ""
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputMarkdown) | Out-Null
$lines | Set-Content -Path $OutputMarkdown -Encoding UTF8
Write-Host "Wrote recall comparison to $OutputMarkdown"

if ($missing.Count -gt $AllowedFalseNegatives -or $changed.Count -gt 0) {
  throw "Recall comparison failed: missing=$($missing.Count), changed=$($changed.Count)."
}
