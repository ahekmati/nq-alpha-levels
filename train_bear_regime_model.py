import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
DAILY_FILE = BASE_DIR / "mnq_daily_research_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

MIN_FEATURE_NON_NULL = 0.80
TOP_N_FEATURES = 20

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
# ---------------------------------------- #


def color(text, c):
    return f"{c}{text}{ANSI_RESET}"


def section(title):
    print()
    print(color(f"{'=' * 12} {title} {'=' * 12}", ANSI_BOLD + ANSI_CYAN))


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def load_data():
    if not DAILY_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DAILY_FILE}")

    df = pd.read_csv(DAILY_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["target_active_bear"] = (d["label"] == "ACTIVE_BEAR").astype(int)

    d["target_exhaustion_vs_active"] = np.where(
        d["label"].isin(["ACTIVE_BEAR", "EXHAUSTION"]),
        (d["label"] == "EXHAUSTION").astype(int),
        np.nan
    )

    d["target_prebear_vs_outside"] = np.where(
        d["label"].isin(["PRE_BEAR", "OUTSIDE"]),
        (d["label"] == "PRE_BEAR").astype(int),
        np.nan
    )

    return d


def choose_features(df: pd.DataFrame) -> list:
    exclude = {
        "time", "label", "bear_window_id",
        "target_active_bear", "target_exhaustion_vs_active", "target_prebear_vs_outside"
    }

    candidate_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null_ratio = df[col].notna().mean()
            if non_null_ratio >= MIN_FEATURE_NON_NULL:
                candidate_cols.append(col)

    return sorted(candidate_cols)


def compute_effect_size_table(df: pd.DataFrame, feature_cols: list, label_col: str = "label") -> pd.DataFrame:
    rows = []

    labels = [x for x in sorted(df[label_col].dropna().unique())]
    if len(labels) < 2:
        return pd.DataFrame()

    for col in feature_cols:
        x = df[[col, label_col]].dropna()
        if x.empty:
            continue

        overall_std = x[col].std(ddof=0)
        if pd.isna(overall_std) or overall_std == 0:
            continue

        means = x.groupby(label_col)[col].mean().to_dict()
        medians = x.groupby(label_col)[col].median().to_dict()

        if "ACTIVE_BEAR" in means and "OUTSIDE" in means:
            active_vs_outside = (means["ACTIVE_BEAR"] - means["OUTSIDE"]) / overall_std
        else:
            active_vs_outside = np.nan

        if "EXHAUSTION" in means and "ACTIVE_BEAR" in means:
            exhaustion_vs_active = (means["EXHAUSTION"] - means["ACTIVE_BEAR"]) / overall_std
        else:
            exhaustion_vs_active = np.nan

        if "PRE_BEAR" in means and "OUTSIDE" in means:
            prebear_vs_outside = (means["PRE_BEAR"] - means["OUTSIDE"]) / overall_std
        else:
            prebear_vs_outside = np.nan

        rows.append({
            "feature": col,
            "mean_active_vs_outside_effect": active_vs_outside,
            "mean_exhaustion_vs_active_effect": exhaustion_vs_active,
            "mean_prebear_vs_outside_effect": prebear_vs_outside,
            "overall_std": overall_std,
            **{f"mean_{k}": v for k, v in means.items()},
            **{f"median_{k}": v for k, v in medians.items()},
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_active_vs_outside"] = out["mean_active_vs_outside_effect"].abs()
    out["abs_exhaustion_vs_active"] = out["mean_exhaustion_vs_active_effect"].abs()
    out["abs_prebear_vs_outside"] = out["mean_prebear_vs_outside_effect"].abs()

    return out.sort_values("abs_active_vs_outside", ascending=False)


def build_models():
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return {
        "logistic_regression": lr,
        "random_forest": rf,
    }


def evaluate_binary_problem(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    problem_name: str,
    positive_label_name: str
):
    section(f"Training {problem_name}")

    work = df.copy()
    work = work.dropna(subset=[target_col]).reset_index(drop=True)

    if work.empty:
        print(color("No rows available for this target.", ANSI_RED))
        return None, None, None

    X = work[feature_cols].copy()
    y = work[target_col].astype(int).copy()

    print(color(f"Rows: {len(work)}", ANSI_YELLOW))
    print(color(f"Positive rate ({positive_label_name}): {100 * y.mean():.2f}%", ANSI_YELLOW))

    models = build_models()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    all_results = []
    all_predictions = []

    for model_name, model in models.items():
        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)

            fold_result = {
                "problem": problem_name,
                "model": model_name,
                "fold": fold,
                "rows_train": len(train_idx),
                "rows_test": len(test_idx),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "auc": safe_auc(y_test, prob),
            }
            fold_rows.append(fold_result)

            fold_preds = pd.DataFrame({
                "time": work.iloc[test_idx]["time"].values,
                "label": work.iloc[test_idx]["label"].values,
                "target": y_test.values,
                "pred_prob": prob,
                "pred_label": pred,
                "problem": problem_name,
                "model": model_name,
                "fold": fold,
            })
            all_predictions.append(fold_preds)

        model_results = pd.DataFrame(fold_rows)
        if model_results.empty:
            continue

        mean_row = {
            "problem": problem_name,
            "model": model_name,
            "fold": "mean",
            "rows_train": model_results["rows_train"].mean(),
            "rows_test": model_results["rows_test"].mean(),
            "accuracy": model_results["accuracy"].mean(),
            "precision": model_results["precision"].mean(),
            "recall": model_results["recall"].mean(),
            "f1": model_results["f1"].mean(),
            "auc": model_results["auc"].mean(),
        }

        all_results.append(model_results)
        all_results.append(pd.DataFrame([mean_row]))

    if not all_results:
        print(color("No valid folds were available.", ANSI_RED))
        return None, None, None

    results_df = pd.concat(all_results, ignore_index=True)
    preds_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    mean_only = results_df[results_df["fold"] == "mean"].copy()
    mean_only = mean_only.sort_values(["f1", "auc"], ascending=False)

    print(color("Model summary:", ANSI_BOLD + ANSI_GREEN))
    for _, row in mean_only.iterrows():
        print(
            f"- {row['model']}: "
            f"F1={fmt(row['f1'])} | "
            f"AUC={fmt(row['auc'])} | "
            f"Precision={fmt(row['precision'])} | "
            f"Recall={fmt(row['recall'])}"
        )

    best_model_name = mean_only.iloc[0]["model"]
    best_model = models[best_model_name]

    best_model.fit(X, y)

    feature_importance_df = None

    if best_model_name == "logistic_regression":
        coef = best_model.named_steps["model"].coef_[0]
        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.abs(coef),
            "signed_coef": coef,
            "model": best_model_name,
            "problem": problem_name,
        }).sort_values("importance", ascending=False)

    elif best_model_name == "random_forest":
        imp = best_model.named_steps["model"].feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": imp,
            "model": best_model_name,
            "problem": problem_name,
        }).sort_values("importance", ascending=False)

    out_results = OUTPUT_DIR / f"{problem_name.lower()}_cv_results.csv"
    out_preds = OUTPUT_DIR / f"{problem_name.lower()}_predictions.csv"
    out_imp = OUTPUT_DIR / f"{problem_name.lower()}_feature_importance.csv"

    results_df.to_csv(out_results, index=False)
    if not preds_df.empty:
        preds_df.to_csv(out_preds, index=False)
    if feature_importance_df is not None:
        feature_importance_df.to_csv(out_imp, index=False)

    print(color(f"Saved CV results to {out_results}", ANSI_CYAN))
    print(color(f"Saved predictions to {out_preds}", ANSI_CYAN))
    print(color(f"Saved feature importance to {out_imp}", ANSI_CYAN))

    return results_df, preds_df, feature_importance_df


def print_top_features(effect_df: pd.DataFrame, feature_importance_df: pd.DataFrame, title: str):
    section(title)

    if effect_df is not None and not effect_df.empty:
        print(color("Top effect-size features:", ANSI_BOLD + ANSI_MAGENTA))
        cols = [
            "feature",
            "mean_active_vs_outside_effect",
            "mean_exhaustion_vs_active_effect",
            "mean_prebear_vs_outside_effect",
        ]
        print(
            effect_df[cols]
            .head(TOP_N_FEATURES)
            .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        )

    if feature_importance_df is not None and not feature_importance_df.empty:
        print()
        print(color("Top model features:", ANSI_BOLD + ANSI_GREEN))
        cols = [c for c in ["feature", "importance", "signed_coef"] if c in feature_importance_df.columns]
        print(
            feature_importance_df[cols]
            .head(TOP_N_FEATURES)
            .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        )


def print_label_distribution(df: pd.DataFrame):
    section("Label Distribution")
    counts = df["label"].value_counts(dropna=False)
    for label, count in counts.items():
        pct = 100 * count / len(df)
        c = ANSI_RED if label == "ACTIVE_BEAR" else ANSI_GREEN if label == "RECOVERY" else ANSI_YELLOW
        print(color(f"{label:>12}: {count:5d} ({pct:6.2f}%)", c))


def print_feature_ideas():
    section("Research Notes")
    print(color("Most promising use cases:", ANSI_BOLD + ANSI_CYAN))
    print("- PRE_BEAR vs OUTSIDE: early warning / arm the bear model.")
    print("- ACTIVE_BEAR vs REST: broad regime detection for whether short logic is allowed.")
    print("- EXHAUSTION vs ACTIVE_BEAR: stop pressing shorts and start watching reversal risk.")

    print(color("Modeling guidance:", ANSI_BOLD + ANSI_YELLOW))
    print("- Use daily model for regime permission.")
    print("- Use H1 later for entry timing inside permitted bear regimes.")
    print("- Favor probability thresholds over raw hard labels for trading logic.")
    print("- Review false positives near sharp corrections that did not evolve into sustained bear legs.")


def main():
    df = load_data()
    df = build_targets(df)
    feature_cols = choose_features(df)

    section("Dataset")
    print(color(f"Rows: {len(df)}", ANSI_YELLOW))
    print(color(f"Feature count: {len(feature_cols)}", ANSI_YELLOW))
    print(color(f"Date range: {df['time'].min()} -> {df['time'].max()}", ANSI_YELLOW))

    print_label_distribution(df)

    effect_df = compute_effect_size_table(df, feature_cols)
    effect_out = OUTPUT_DIR / "daily_feature_effect_sizes.csv"
    effect_df.to_csv(effect_out, index=False)
    print(color(f"\nSaved effect-size table to {effect_out}", ANSI_CYAN))

    active_results, active_preds, active_imp = evaluate_binary_problem(
        df=df,
        feature_cols=feature_cols,
        target_col="target_active_bear",
        problem_name="ACTIVE_BEAR",
        positive_label_name="ACTIVE_BEAR"
    )

    exhaustion_results, exhaustion_preds, exhaustion_imp = evaluate_binary_problem(
        df=df,
        feature_cols=feature_cols,
        target_col="target_exhaustion_vs_active",
        problem_name="EXHAUSTION_VS_ACTIVE",
        positive_label_name="EXHAUSTION"
    )

    prebear_results, prebear_preds, prebear_imp = evaluate_binary_problem(
        df=df,
        feature_cols=feature_cols,
        target_col="target_prebear_vs_outside",
        problem_name="PREBEAR_VS_OUTSIDE",
        positive_label_name="PRE_BEAR"
    )

    print_top_features(effect_df, active_imp, "Feature Ranking")
    print_feature_ideas()

    section("Next Step")
    print("1. Use the ACTIVE_BEAR probability as the daily regime filter.")
    print("2. Use the EXHAUSTION probability as the stop-shorting / disarm signal.")
    print("3. Build a separate H1 entry model only inside days where ACTIVE_BEAR probability is high.")
    print("4. Review the saved prediction CSVs around each labeled episode and inspect false positives.")

if __name__ == "__main__":
    main()
