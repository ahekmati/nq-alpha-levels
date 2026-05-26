# NQ Alpha Levels Bot

A machine learning system that identifies key support levels on @MNQ (Micro Nasdaq Futures) and sends Telegram alerts when high-confidence bounce setups are detected.

---

## How It Works

The system uses an XGBoost classifier trained on historical support level retests. It learns which combinations of features (HTF trend, RSI, session, level age, touch count, etc.) predict a successful bounce off a support level.

**Entry logic:**
- Daily close must be above the Daily 100 EMA (bullish trend only)
- Price must be within a configurable ATR distance of a known support level
- ML model score must be ≥ 60% (watch) or ≥ 75% (strong)

**Trade parameters (from backtest study):**
- Entry: limit order at the level price
- Stop loss: 1.0 × ATR below the level
- Take profit: 2.0R above entry
- Backtest results (2019-2026): 57-61% win rate, PF 2.3-2.8

---

## Files

| File | Purpose | Runs on |
|---|---|---|
| `nq_alpha_levels_bot.py` | Alert scanner — sends Telegram signals | Linux (cron) |
| `levels_ml.py` | Research tool — collect, train, validate, backtest, retrain | Linux (manual) |
| `levels_bot.py` | Live trading bot (future use) | Windows VPS |
| `levels_xgb_model.joblib` | Trained XGBoost model | Both |
| `levels_dataset.csv` | Labeled retest dataset | Linux |
| `retrain_baseline.json` | Baseline AUC for monthly comparison | Linux |
| `retrain_report.json` | Audit trail of last retrain | Linux |

**Do not push to GitHub:**
- `bot_state.json` — live trade state
- `backtest_trades.csv` — trade log
- `study_results.csv` — study output
- `logs/` — log files
- `.env` — credentials (if used)

---

## Setup

### Requirements

```bash
pip install mt5linux xgboost scikit-learn pandas numpy joblib requests
```

### Telegram Configuration

Open `nq_alpha_levels_bot.py` and set at the top:

```python
TELEGRAM_TOKEN   = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

To create a Telegram bot:
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the token it gives you
4. Start a chat with your bot, then visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Send a message to the bot and refresh — your chat ID will appear

### MT5 Configuration

The alert script connects via mt5linux. MT5 terminal must be running on your machine. Default connection:

```python
MT5_HOST = "127.0.0.1"
MT5_PORT = 18812
```

---

## Daily Usage

### The cron job runs automatically

Every 30 minutes, Monday-Friday, 08:00-21:00 UTC:

```
*/30 8-21 * * 1-5 cd /path/to/project && .venv/bin/python nq_alpha_levels_bot.py >> logs/alert_cron.log 2>&1
```

You do nothing. Just wait for Telegram to ping you.

### When you receive an alert

```
🎯 LEVEL ALERT — MNQM26 H1
🕐 2026-05-28 14:30 UTC
💲 Price: 29,868.50  |  ATR: 89.5 pts
📈 Daily trend: BULLISH ✅

🔥 STRONG (75%+)
  📍 29,743.00 | score 84% | 2.4 ATR away | 3 touches | age 131bars
     Entry: 29,743.00  SL: 29,654.00  TP: 29,921.00
     Risk: 89pts ($178)  Target: 178pts ($356)

📋 Last signals alerted:
  2026-05-26 14:30 | 29,113.00 | 71% | ⏳ pending
```

**What to do:**
1. Open your chart and look at the level visually
2. Does it show prior bounces? Does it look like a logical support?
3. If yes — place a **buy limit order** in MT5:
   - Price: the Entry shown
   - Stop Loss: the SL shown
   - Take Profit: the TP shown
   - Volume: 1 MNQ contract
4. If the chart looks messy or price has already moved too far — skip it

### Alert score tiers

| Score | Label | Action |
|---|---|---|
| Below 60% | Not alerted | Log only |
| 60% – 74% | 👀 WATCH | Worth checking chart |
| 75%+ | 🔥 STRONG | High confidence — prioritize |

### After a trade resolves

Mark the outcome to build your live track record:

```bash
# If the trade hit take profit
python nq_alpha_levels_bot.py --update 29743 win

# If the trade hit stop loss
python nq_alpha_levels_bot.py --update 29743 loss

# If you cancelled / didn't take it
python nq_alpha_levels_bot.py --update 29743 cancelled
```

### View your signal history

```bash
python nq_alpha_levels_bot.py --history 20
```

Output:
```
LAST 20 SIGNALS FROM HISTORY LOG
  Timestamp            Level    Score   Dist  Touches   Entry      SL        TP     Outcome
  2026-05-28 14:30  29,743.00   84%    2.4     3     29,743.00  29,654.00  29,921.00  ✅ win
  2026-05-26 09:00  29,113.00   71%    6.2     2     29,113.00  29,024.00  29,291.00  ⏳ pending

  Resolved: 1  (1W / 0L)  WR: 100%  Net R: +1.5R  Pending: 1
```

### Test Telegram connection

```bash
python nq_alpha_levels_bot.py --test
```

---

## Research Tool — levels_ml.py

All research and model maintenance is done via `levels_ml.py`.

### Modes

```bash
# Pull historical bars and build labeled dataset
python levels_ml.py --mode collect --symbol @MNQ --tf H1 --bars 50000

# Train XGBoost model on dataset
python levels_ml.py --mode train

# Walk-forward OOS validation
python levels_ml.py --mode validate --threshold 0.70

# SL/TP optimization study (bar-by-bar simulation)
python levels_ml.py --mode study --threshold 0.65

# Bar-by-bar backtest (identical to live bot logic)
python levels_ml.py --mode backtest --symbol @MNQ --tf H1 --bars 20000 \
  --threshold 0.70 --sl-atr 1.0 --tp-r 2.0

# Scan current levels with ML scores
python levels_ml.py --mode scan --symbol @MNQ --tf H1

# Full retrain pipeline with AUC check (run monthly)
python levels_ml.py --mode retrain --symbol @MNQ --tf H1 --bars 50000
```

---

## Monthly Maintenance

Run on the **first Monday of every month**:

```bash
python levels_ml.py --mode retrain --symbol @MNQ --tf H1 --bars 50000
```

**If output shows:**
```
✅ Model updated — copy levels_xgb_model.joblib to VPS and restart bot
```
→ Nothing else needed. The alert script picks up the new model automatically on the next cron run.

**If output shows:**
```
⚠️  Model NOT updated — AUC dropped...
```
→ Keep running with current model. Check again in 2 weeks.

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost classifier |
| Training samples | 2,263 labeled retests |
| Date range | 2019-2026 |
| OOS AUC | 0.7047 |
| OOS win rate @ 70% | 61.1% |
| OOS profit factor @ 70% | 2.357 |
| OOS expectancy @ 70% | +0.528R/trade |
| Top features | htf_trend, pct_from_ema50, htf_pct_ema20 |

### Feature categories

- **Level quality:** touch count, age, departure height, precision
- **Approach:** consecutive red bars, volume ratio, drop distance
- **Retest candle:** wick/body ratio, close position, body size
- **Momentum:** RSI, distance from EMA 20/50/200
- **HTF context:** H4 trend, H4 EMA slope, confluence
- **Session:** hour, London/NY/Asia, day of week
- **Structure:** distance to round number

---

## Backtest Summary

Tested on @MNQ H1, 2023-2026, 20,000 bars:

| Parameter | Value |
|---|---|
| ML threshold | 70% |
| SL | 1.0 × ATR |
| TP | 2.0R |
| Total trades | 247 |
| Win rate | 57.1% |
| Profit factor | 2.462 |
| Net R | +152.26R |
| Net P&L (1 MNQ) | +$44,779 |
| Max drawdown | $-1,942 |
| Avg bars held | 15.7 |

---

## Contract Roll

The system auto-detects the front month MNQ contract by checking volume across all `MNQ*` symbols. No manual update needed on quarterly rolls (March, June, September, December).

---

## Troubleshooting

**No alerts for several days**
Normal when price is extended away from key levels. Check the cron log:
```bash
grep "SCAN START" logs/alert_cron.log | tail -5
```
You should see recent timestamps confirming the cron is running.

**MT5 connection error**
MT5 terminal must be open and running. Check mt5linux server is active.

**Score seems low on all levels**
Levels with only 1 touch and age < 20 bars will always score low. The model needs proven levels. Wait for price to form and retest meaningful structure.

**Model feels stale**
Run the monthly retrain early:
```bash
python levels_ml.py --mode retrain --symbol @MNQ --tf H1 --bars 50000
```

---

## Future — Live Bot Deployment

After 2-3 months of manual trading and signal logging, review your live track record:

```bash
python nq_alpha_levels_bot.py --history 50
```

If win rate and profit factor are consistent with backtest results, deploy `levels_bot.py` to the Windows VPS for automated execution. See `levels_bot.py` header for VPS setup instructions.

---

## File Structure

```
project/
  nq_alpha_levels_bot.py    ← alert script (cron)
  levels_ml.py              ← research tool
  levels_bot.py             ← live bot (future)
  levels_xgb_model.joblib   ← trained model
  levels_dataset.csv        ← labeled dataset
  retrain_baseline.json     ← AUC baseline
  retrain_report.json       ← retrain audit trail
  logs/
    alert_cron.log          ← cron output
    signal_history.json     ← live signal log
    last_alert.json         ← duplicate suppression
    retrain.log             ← retrain cron output
```
