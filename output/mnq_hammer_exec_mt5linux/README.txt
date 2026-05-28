MNQ Hammer Execution Backtest via mt5linux

Symbol: @MNQ
Timeframe: H1
Hammer spec: standard
Entry mode: next_open
Stop mode: hammer_low_buffer
Target mode: r_multiple
Date from: 2018-01-01
Date to: now
mt5linux host: 127.0.0.1
mt5linux port: 18812

Files:
- trades.csv: executed trade log
- equity_curve.csv: equity curve by exit timestamp
- summary.csv: overall results
- yearly_stats.csv: year-by-year breakdown

Prereqs:
- Install mt5linux on Linux.
- Install MetaTrader5 and mt5linux in the Wine Windows Python.
- Run the mt5linux server under Wine while MT5 is open.
