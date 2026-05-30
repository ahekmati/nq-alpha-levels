# NQ Alpha Overnight Strategy

MNQ overnight dip-buying strategy based on two validated setups:

- **Study 1**: Buy overnight dip after strong RTH trend day
- **Study 2**: Buy overnight dip after large overnight rally + RTH selloff

Backtested on 7 years of @MNQ H1 data (2019–2026).  
Executes live via MT5 on Windows VPS.

---

## Project Structure

```
nq-alpha-overnight/
├── strategy/
│   ├── evaluator.py          # Runs at 4PM ET — evaluates tonight's setup
│   └── watcher.py            # Runs overnight — monitors price + executes orders
├── ml/
│   └── study2_ml_filter.py   # XGBoost gate for Study 2 (optional)
├── logs/                     # Auto-created — evaluator.log, watcher.log, trades.log
├── tonight_setup.json        # Auto-created by evaluator — gitignored
├── run_overnight.bat         # Windows launcher
└── requirements.txt
```

---

## Validated Edge (Backtested 2019–2026)

### Study 1 — Strong Up Day Overnight Dip
| Param | Value |
|---|---|
| Entry | RTH close − 0.75x ATR |
| Stop | Entry − 1.0x ATR |
| Target | Entry + 2.0R |
| Win rate | 75% (44 trades / 7 years) |
| Expectancy | +1.25R |
| Trigger rate | 37% of qualifying days |

Qualifying day filters:
- Daily RSI(10) ≥ 60
- H1 close > H1 EMA(100)
- H1 EMA(20) > H1 EMA(100)
- RTH session gain ≥ 0.8%

### Study 2 — Overnight Rally → RTH Selloff → Overnight Dip
| Param | Value |
|---|---|
| Entry | RTH close − 2.0x ATR |
| Stop | Entry − 1.0x ATR |
| Target | Entry + 2.5R |
| Win rate | 49–56% (800 setups / 7 years) |
| Expectancy | +0.95R |
| Reversal rate | 87% (price recovers overnight) |

Qualifying setup filters:
- Prior overnight rally ≥ 2.5x ATR
- RTH session retraced ≥ 0.5% from overnight high

Best entry session: Asian (16:00–00:00 ET) — 55.9% WR

---

## Windows VPS Deployment

### 1. Clone into existing repo

```bash
# On Windows VPS, in your repo directory
git pull origin main
```

The `nq-alpha-overnight/` folder should appear.

### 2. Set up Python environment

```bash
cd nq-alpha-overnight
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure the .bat launcher

Edit `run_overnight.bat` and update the `VENV` path:
```bat
SET VENV=C:\path\to\nq-alpha-overnight\.venv\Scripts\activate.bat
```

### 4. Schedule via Windows Task Scheduler

- Open Task Scheduler → Create Basic Task
- Name: `NQ Alpha Overnight`
- Trigger: Daily at **16:05 ET**
- Action: Start program → `C:\path\to\run_overnight.bat`
- Settings: Run whether user logged in or not ✓

### 5. First run — paper mode

Before going live, validate the evaluator output for a few days:

```bash
python strategy/evaluator.py
cat tonight_setup.json
```

Check that setup conditions are triggering correctly, entry/stop/target
levels look reasonable, and the JSON structure is valid.

### 6. Run ML filter (optional)

First copy your `study2_all_trades.csv` from the Linux study:

```bash
# From Linux:
scp ~/projects/mt5-python/mnq_study/study2_all_trades.csv user@vps:path/mnq_study/

# Then train:
python ml/study2_ml_filter.py --mode train --data ./mnq_study/study2_all_trades.csv
```

If AUC > 0.58 and win rate lift > 5pp, model is saved automatically.
Uncomment the scan step in `run_overnight.bat` to enable.

---

## Log Files

| File | Contents |
|---|---|
| `logs/evaluator.log` | Daily condition evaluation + armed setup details |
| `logs/watcher.log` | Overnight price monitoring + order placement |
| `logs/trades.log` | Trade entries, fills, closes + P&L |

All logs are appended (not overwritten) — review weekly.

---

## Pushing updates from Linux dev machine

```bash
# On Linux (~/projects/mt5-python):
git remote add origin https://github.com/ahekmati/nq-alpha-levels.git
git add nq-alpha-overnight/
git commit -m "add overnight dip strategy v1"
git push origin main

# On Windows VPS:
git pull origin main
```

---

## Monitoring

Check `logs/trades.log` each morning. Key things to review:

1. Did the evaluator fire correctly at 4PM?
2. Was a setup armed? (if not, was the reason valid?)
3. If armed, did the watcher place an order?
4. Was the order filled? What was the result?
5. Was EOD close triggered at 09:15 if no TP/SL hit?

After 20+ live trades, compare live win rates to backtest.
If Study 2 live WR > 50%, ML filter likely won't add value.
If Study 2 live WR < 45%, re-examine setup conditions.
