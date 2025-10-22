@echo off
echo =====================================================
echo       CREDIT CARD FRAUD DETECTION PROJECT
echo =====================================================
echo.
echo Activating virtual environment...
echo.

call activate.bat

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo Please ensure activate.bat exists in the current directory
    pause
    exit /b 1
)

echo [OK] Virtual environment activated successfully
echo.
echo Starting project...
echo.

python run_project.py

echo.
echo Project finished.
pause