@echo off
cd /d C:\Users\Administrator\mt5scraper
.venv\Scripts\python.exe levels_ml_win.py --mode retrain --symbol "@MNQ" --tf H1 --bars 50000 >> logs\retrain.log 2>&1
