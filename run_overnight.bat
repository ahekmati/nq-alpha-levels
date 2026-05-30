@echo off
REM ============================================================
REM  NQ Alpha Overnight Strategy Launcher
REM  Runs evaluator at 4PM ET, then starts watcher for overnight
REM
REM  Schedule this .bat via Windows Task Scheduler at 16:05 ET
REM  Or run manually after RTH close
REM ============================================================

SET VENV=C:\path\to\your\venv\Scripts\activate.bat
SET STRATEGY_DIR=%~dp0strategy

echo [%DATE% %TIME%] Starting NQ Alpha Overnight...

REM Activate virtual environment
call %VENV%

REM Step 1: Run evaluator — determines tonight's setup
echo [%DATE% %TIME%] Running evaluator...
python "%STRATEGY_DIR%\evaluator.py"
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] Evaluator failed — check logs\evaluator.log
    pause
    exit /b 1
)

REM Step 2: (Optional) Score with ML filter
REM Uncomment once model is trained:
REM python "%~dp0ml\study2_ml_filter.py" --mode scan --setup "%~dp0tonight_setup.json"

REM Step 3: Start watcher overnight
echo [%DATE% %TIME%] Starting overnight watcher...
python "%STRATEGY_DIR%\watcher.py"

echo [%DATE% %TIME%] Strategy session complete.
pause
