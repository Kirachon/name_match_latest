#requires -Version 5.0
<#
.SYNOPSIS
    Start the repo-owned MySQL fixture and seed deterministic smoke data.

.DESCRIPTION
    Uses docker compose to start the repo-scoped MySQL fixture on
    127.0.0.1:3307, waits for health, then runs the Rust seed helper against
    the duplicate_checker DB.

.PARAMETER Rows
    Number of sample rows per table to seed.

.PARAMETER HostPort
    Host port to publish MySQL on. Defaults to 3307.

.PARAMETER KeepRunning
    Leave the Docker fixture running after the smoke setup.
#>
param(
  [int]$Rows = 1000,
  [int]$HostPort = 3307,
  [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
  $env:MATCHERS_MYSQL_PORT = [string]$HostPort
  docker compose up -d matchers-mysql
  if ($LASTEXITCODE -ne 0) { throw "docker compose up failed with exit code $LASTEXITCODE" }

  $containerId = docker compose ps -q matchers-mysql
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($containerId)) {
    throw "Unable to resolve docker compose container id for matchers-mysql"
  }

  $deadline = (Get-Date).AddMinutes(3)
  do {
    $health = docker inspect $containerId --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}"
    if ($LASTEXITCODE -ne 0) { throw "docker inspect failed with exit code $LASTEXITCODE" }
    if ($health -eq "healthy" -or $health -eq "running") { break }
    Start-Sleep -Seconds 2
  } while ((Get-Date) -lt $deadline)

  if ($health -ne "healthy" -and $health -ne "running") {
    throw "matchers-mysql did not become ready; final health=$health"
  }

  if (-not $env:CARGO_HOME) {
    $env:CARGO_HOME = "C:\cargo_nm_temp"
  }

  cargo run --locked --bin seed -- `
    127.0.0.1 `
    $HostPort `
    root `
    root `
    duplicate_checker `
    sample_a `
    sample_b `
    $Rows

  if ($LASTEXITCODE -ne 0) { throw "cargo seed failed with exit code $LASTEXITCODE" }

  Write-Host "Docker MySQL smoke fixture is ready on 127.0.0.1:$HostPort"
}
finally {
  if (-not $KeepRunning) {
    docker compose stop matchers-mysql | Out-Null
  }
  Pop-Location
}
