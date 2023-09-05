@echo off
rem Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and try again.
    pause
    exit /b 1
)

rem Update pip packages
echo Updating pip packages...
python -m pip install -r requirements.txt

rem Execute your Python script
echo Executing your Python script...
python start_x.py

pause