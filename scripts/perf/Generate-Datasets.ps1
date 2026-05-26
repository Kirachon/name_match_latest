<#
.SYNOPSIS
    Generate deterministic CSV datasets for name matcher performance baselines.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\perf\Generate-Datasets.ps1
#>
param(
  [string]$OutputDir = "tmp/perf/datasets",
  [int]$Seed = 424242,
  [int]$SmallRows = 1000,
  [int]$MediumRows = 100000,
  [int]$LargeRows = 1000000,
  [switch]$SkipLarge
)

$ErrorActionPreference = "Stop"

function New-DeterministicPerson {
  param(
    [int]$Index,
    [int]$SeedValue,
    [string]$Profile,
    [string]$Side
  )

  $firstNames = @("Ana", "Maria", "Juan", "Jose", "Carla", "Ramon", "Liza", "Mark", "Grace", "Paolo")
  $lastNames = @("Santos", "Reyes", "Cruz", "Garcia", "Mendoza", "Bautista", "Torres", "Ramos", "Flores", "Rivera")
  $middleNames = @("", "Dela", "Mae", "Ann", "Luis", "Marie", "Jose", "Luz")

  $first = $firstNames[($Index + $SeedValue) % $firstNames.Count]
  $last = $lastNames[(($Index * 7) + $SeedValue) % $lastNames.Count]
  $middle = $middleNames[(($Index * 3) + $SeedValue) % $middleNames.Count]
  $year = 1970 + (($Index + $SeedValue) % 35)
  $month = 1 + (($Index * 5) % 12)
  $day = 1 + (($Index * 11) % 28)

  if ($Profile -eq "high_collision_birthdate") {
    $year = 1985
    $month = 1 + ($Index % 2)
    $day = 15
  } elseif ($Profile -eq "high_collision_birth_year") {
    $year = 1990
  } elseif ($Profile -eq "l10_l11_fuzzy_heavy") {
    $year = 1980 + ($Index % 5)
    if ($Side -eq "target" -and ($Index % 4) -eq 0) {
      $first = "$first" + "h"
    }
    if ($Side -eq "target" -and ($Index % 5) -eq 0) {
      $last = "$last" + "s"
    }
  }

  [pscustomobject]@{
    id = if ($Side -eq "source") { $Index + 1 } else { $Index + 100000001 }
    uuid = "00000000-0000-0000-0000-{0:D12}" -f $Index
    first_name = $first
    middle_name = $middle
    last_name = $last
    birthdate = "{0:D4}-{1:D2}-{2:D2}" -f $year, $month, $day
    hh_id = "HH-{0:D8}" -f [int][math]::Floor($Index / 4)
    barangay_code = "BRGY-{0:D3}" -f ($Index % 128)
  }
}

function Write-Dataset {
  param(
    [string]$Name,
    [int]$Rows,
    [string]$Profile
  )

  $dir = Join-Path $OutputDir $Name
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $source = Join-Path $dir "source.csv"
  $target = Join-Path $dir "target.csv"

  0..($Rows - 1) |
    ForEach-Object { New-DeterministicPerson -Index $_ -SeedValue $Seed -Profile $Profile -Side "source" } |
    Export-Csv -Path $source -NoTypeInformation -Encoding UTF8

  0..($Rows - 1) |
    ForEach-Object { New-DeterministicPerson -Index $_ -SeedValue ($Seed + 17) -Profile $Profile -Side "target" } |
    Export-Csv -Path $target -NoTypeInformation -Encoding UTF8

  $sourceHash = (Get-FileHash -Algorithm SHA256 -Path $source).Hash.ToLowerInvariant()
  $targetHash = (Get-FileHash -Algorithm SHA256 -Path $target).Hash.ToLowerInvariant()
  $manifest = [ordered]@{
    name = $Name
    profile = $Profile
    seed = $Seed
    rows_per_side = $Rows
    source = (Resolve-Path $source).Path
    target = (Resolve-Path $target).Path
    source_sha256 = $sourceHash
    target_sha256 = $targetHash
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  }
  $manifest | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $dir "manifest.json") -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Dataset -Name "small_csv_1k" -Rows $SmallRows -Profile "balanced"
Write-Dataset -Name "high_collision_birthdate" -Rows $SmallRows -Profile "high_collision_birthdate"
Write-Dataset -Name "high_collision_birth_year" -Rows $SmallRows -Profile "high_collision_birth_year"
Write-Dataset -Name "l10_l11_fuzzy_heavy" -Rows $SmallRows -Profile "l10_l11_fuzzy_heavy"
Write-Dataset -Name "medium_csv_100k" -Rows $MediumRows -Profile "balanced"

if (-not $SkipLarge) {
  Write-Dataset -Name "large_csv_1m" -Rows $LargeRows -Profile "balanced"
}

Write-Host "Generated datasets under $OutputDir"
