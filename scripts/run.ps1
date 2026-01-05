# scripts/run.ps1 - PowerShell version for Windows
# Note: X11 forwarding on Windows requires VcXsrv or similar

Write-Host "Starting AFPRS containers..." -ForegroundColor Green
Write-Host "Note: For GUI on Windows, install VcXsrv and set DISPLAY variable" -ForegroundColor Yellow

# Set DISPLAY for WSL2 (adjust IP as needed)
$env:DISPLAY = "host.docker.internal:0.0"

docker-compose up
