# scripts/build.ps1 - PowerShell version for Windows
Write-Host "Building AFPRS containers..." -ForegroundColor Green
docker-compose build
Write-Host "Done!" -ForegroundColor Green
