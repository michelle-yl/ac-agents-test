# Apply monitoring_events schema on Windows (PowerShell-safe).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$sql = Join-Path $PSScriptRoot "..\schema\002_monitoring_events.sql"
if (-not (Test-Path $sql)) {
    Write-Error "Missing $sql"
}
Get-Content $sql | docker compose exec -T postgres psql -U angie -d angie_monitoring_replica
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Applied 002_monitoring_events.sql"
