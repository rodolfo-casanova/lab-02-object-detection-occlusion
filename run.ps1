# Lab 02 - Small and occluded object detection
# Windows PowerShell version of run.sh. Run it from this folder:
#   powershell -ExecutionPolicy Bypass -File .\run2.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install Docker Desktop and start it, then run this script again."
}

# PowerShell does not stop when a native command fails, so every docker call checks its exit code.
function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Invoke-Docker compose build
Invoke-Docker compose run --rm detection

Write-Host ""
Write-Host "Done. Metrics file: results\detection\metrics.json"
