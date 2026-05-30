@echo off
cd /d C:\Users\Administrator\mt5scraper
.venv\Scripts\python.exe supertrend_ml_win.py --live --retrain --no-debug >> logs\supertrend_live.log 2>&1
