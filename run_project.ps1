Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "       CREDIT CARD FRAUD DETECTION PROJECT        " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
Write-Host ""

try {
    # Activate the virtual environment
    & ".\Activate.ps1"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate virtual environment"
    }

    Write-Host "[OK] Virtual environment activated successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting project..." -ForegroundColor Yellow
    Write-Host ""

    # Run the Python script
    python run_project.py

    Write-Host ""
    Write-Host "Project finished." -ForegroundColor Cyan
    Read-Host "Press Enter to continue"
}
catch {
    Write-Host "[ERROR] $_" -ForegroundColor Red
    Write-Host "Please ensure Activate.ps1 exists in the current directory" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}