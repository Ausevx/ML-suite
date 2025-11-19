@echo off
REM ML-Suite Installation Script for Windows Command Prompt
REM Run this script as Administrator

echo ML-Suite Installation Script
echo ============================
echo.

REM Check if Chocolatey is installed
where choco >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Chocolatey is not installed. Installing Chocolatey...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    echo Chocolatey installed successfully!
    echo Please restart your terminal and run this script again.
    pause
    exit /b
)

REM Install system dependencies
echo.
echo Installing system dependencies...
choco install tesseract -y
choco install poppler -y

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Installing Python...
    choco install python -y
    echo Please restart your terminal and run this script again.
    pause
    exit /b
)

REM Check Python version
echo.
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip and build tools (important for Python 3.13+)
echo.
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

REM Install Python dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo ============================
echo Installation complete!
echo.
echo To run the application:
echo   1. Activate the virtual environment: venv\Scripts\activate.bat
echo   2. Run: python app.py
echo   3. Open http://localhost:5000 in your browser
echo.
pause

