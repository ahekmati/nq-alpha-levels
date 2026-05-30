@echo off
cd /d "C:\Users\Administrator\mt5scraper"

if not exist "C:\Users\Administrator\mt5scraper\history" mkdir "C:\Users\Administrator\mt5scraper\history"
forfiles /p "C:\Users\Administrator\mt5scraper\history" /m *.log /d -7 /c "cmd /c del /q @path" 2>nul
forfiles /p "C:\Users\Administrator\mt5scraper\history" /m *.jsonl /d -14 /c "cmd /c del /q @path" 2>nul

set "NOCOLOR=1"
set "DEBUG=1"
set "MT5SYMBOL="
set "MT5SYMBOLROOT=MNQ"
set "AUTOCONTRACTROLLOVER=1"
set "MT5PATH="
set "MT5LOGIN="
set "MT5PASSWORD="
set "MT5SERVER="
set "MT5TIMEFRAME=H1"
set "CONSENSUSMIN=2"
set "STRONGCONSENSUSMIN=4"
set "BASELOT=1.0"
set "DOUBLELOT=2.0"
set "ALLOWDOUBLESIZE=0"
set "STOPMODE=dynamic"
set "ATRPERIOD=14"
set "STOPPOINTS=200"
set "MAXSLPRICEDISTANCE=200"
set "SLCAPRETRIES=5"
set "SLCAPRETRYDELAY=1.0"
set "TAKEPROFITPOINTS=800"
set "BLOCKDATES="
set "ONELOSSPERDAY=1"
set "BLOCKONANYSYMBOLPOSITION=1"
set "STRATEGYCOMMENTTAG=AMP"
set "ALLOWDIRECTFLIP=0"
set "LOGDIR=C:\Users\Administrator\mt5scraper\history"

set "ENABLE_SESSION_FILTER=1"
set "ALLOWED_UTC_WINDOWS=12-13,22-23"

"C:\Users\Administrator\mt5scraper\.venv\Scripts\python.exe" "C:\Users\Administrator\mt5scraper\mt5scraper.py" > "C:\Users\Administrator\mt5scraper\history\cron_mt5scraper.log" 2>&1