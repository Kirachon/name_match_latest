<#
.SYNOPSIS
    Capture MySQL version and InnoDB settings for scale-gate evidence.

.EXAMPLE
    $env:MYSQL_IMPORT_TEST_URL = "<mysql-url-for-local-test-database>"
    powershell -ExecutionPolicy Bypass -File scripts\perf\Capture-MysqlConfig.ps1 `
      -Url $env:MYSQL_IMPORT_TEST_URL `
      -OutputPath tmp\perf\mysql-config.json
#>
param(
  [string]$Url = $env:MYSQL_IMPORT_TEST_URL,
  [string]$OutputPath = "tmp/perf/mysql-config.json"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Url)) {
  throw "Set -Url or MYSQL_IMPORT_TEST_URL to a local MySQL test database URL"
}

# Parse a mysql URL with user, password, host, optional port, and database.
if ($Url -notmatch '^mysql://([^:]+):([^@]*)@([^:/]+)(?::(\d+))?/([^?]+)') {
  throw "URL must include user, password, host, optional port, and database"
}
$user = $Matches[1]
$pass = $Matches[2]
$host = $Matches[3]
$port = if ($Matches[4]) { $Matches[4] } else { "3306" }
$database = $Matches[5]

$mysqlArgs = @(
  "-h", $host,
  "-P", $port,
  "-u", $user,
  if ($pass) { "-p$pass" }
  "-N", "-B",
  "-e", "SELECT VERSION();"
)

$version = & mysql @mysqlArgs 2>&1
if ($LASTEXITCODE -ne 0) {
  throw "mysql client failed: $version"
}

$varsQuery = @"
SELECT VARIABLE_NAME, VARIABLE_VALUE
FROM performance_schema.global_variables
WHERE VARIABLE_NAME IN (
  'version',
  'version_comment',
  'innodb_buffer_pool_size',
  'local_infile',
  'max_connections',
  'innodb_flush_log_at_trx_commit'
)
UNION ALL
SELECT VARIABLE_NAME, VARIABLE_VALUE
FROM performance_schema.global_variables
WHERE VARIABLE_NAME LIKE 'innodb_buffer%'
   OR VARIABLE_NAME LIKE 'innodb_log%'
ORDER BY 1;
"@

$varLines = & mysql -h $host -P $port -u $user $(if ($pass) { "-p$pass" }) -N -B -D $database -e $varsQuery 2>&1
if ($LASTEXITCODE -ne 0) {
  # Older servers without performance_schema.global_variables layout
  $varLines = & mysql -h $host -P $port -u $user $(if ($pass) { "-p$pass" }) -N -B -e @"
SHOW VARIABLES WHERE Variable_name IN (
  'innodb_buffer_pool_size','local_infile','max_connections','innodb_flush_log_at_trx_commit'
);
"@ 2>&1
}

$variables = [ordered]@{}
foreach ($line in $varLines) {
  if ($line -match '^(\S+)\s+(.+)$') {
    $variables[$Matches[1]] = $Matches[2]
  }
}

$out = [ordered]@{
  captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  connection = [ordered]@{
    host = $host
    port = $port
    database = $database
    user = $user
  }
  version = ($version | Select-Object -First 1).ToString().Trim()
  variables = $variables
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
$out | ConvertTo-Json -Depth 6 | Set-Content -Path $OutputPath -Encoding UTF8
Write-Host "Wrote MySQL config snapshot to $OutputPath"
