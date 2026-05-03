@echo off
REM Quick activation script for Windows
REM Run this batch file to activate the virtual environment and display key info

echo.
echo ============================================================
echo  Taxi Demand Prediction - Environment Activation
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo ✅ Virtual environment activated!
echo.
echo 📦 Quick Commands:
echo   - Verify setup:     python scripts/verify_setup.py
echo   - Ingest data:      python scripts/01_data_ingestion.py
echo   - Validate data:    python scripts/02_data_validation.py
echo   - Run full pipeline: make install
echo   - Run tests:        pytest tests/ -v
echo.
echo 📁 Project Location: %cd%
echo 🐍 Python Version:
python --version
echo.
echo ============================================================
