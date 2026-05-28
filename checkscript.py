python - <<'PY'
import pandas as pd

d1 = pd.read_csv("research_mnq_swing_meanrev/d1_bars.csv")
h1 = pd.read_csv("research_mnq_swing_meanrev/h1_bars.csv")

print("D1 rows:", len(d1), "cols:", list(d1.columns))
print("H1 rows:", len(h1), "cols:", list(h1.columns))
print("D1 first/last:", d1["time"].iloc[0], "->", d1["time"].iloc[-1])
print("H1 first/last:", h1["time"].iloc[0], "->", h1["time"].iloc[-1])
PY
