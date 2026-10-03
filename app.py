"""
Marketing Funnel Analytics
Campaign Performance & Conversion Intelligence

Future Interns - Data Science & Analytics Task 3

Interactive Streamlit dashboard for the UCI Bank Marketing dataset
(Moro, Laureano & Cortez, 2011). Run with:  streamlit run app.py

Terminology used throughout the app
  Observed Data          -> values read directly from the dataset
  Derived Metrics        -> values calculated from observed data (rates, buckets, shares)
  Analytical Associations-> patterns in this dataset; NOT proof of causation
  Recommendations        -> suggested actions derived from the associations
"""
from __future__ import annotations

import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).parent
DATA_CANDIDATES = [
    APP_DIR / "bank-full.csv",
    APP_DIR / "data" / "bank-full.csv",
    APP_DIR / "bank.csv",
    APP_DIR / "data" / "bank.csv",
]
LOGO_PATH = APP_DIR / "assets" / "logo.png"

CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "poutcome"]
NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
TARGET_COL = "y"
EXPECTED_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing",
                    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                    "previous", "poutcome", "y"]

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
AGE_LABELS = ["18–25", "26–35", "36–45", "46–55", "56–65", "66+"]
CAMPAIGN_LABELS = ["1", "2", "3", "4", "5", "6–10", "11+"]
PREVIOUS_LABELS = ["0", "1", "2", "3", "4–5", "6+"]
DURATION_LABELS = ["0–1 min", "1–2 min", "2–3 min", "3–5 min", "5–10 min", "10+ min"]
PDAYS_LABELS = ["Not previously contacted", "≤30 days", "31–90 days", "91–180 days",
                "181–365 days", "365+ days"]

MIN_N = 100  # minimum customers in a segment before it is used in an insight

C_PRIMARY, C_POS, C_NEG, C_WARN, C_MUTED = "#3B82F6", "#14B8A6", "#94A3B8", "#F59E0B", "#64748B"

PAGES = [
    "Executive Dashboard",
    "Funnel Analysis",
    "Channel & Campaign",
    "Customer Segmentation",
    "Campaign Performance",
    "Customer Insights",
    "Recommendations",
    "Predictive Insights",
    "Data Quality & Export",
]

FEATURE_GROUPS = {
    "Customer profile": ["age", "job", "marital", "education", "default", "balance", "housing", "loan"],
    "Contact plan": ["contact", "month", "day"],
    "Contact history": ["campaign", "previous", "pdays", "poutcome"],
}

_PLOTLY_NEW_WIDTH_API = "width" in inspect.signature(st.plotly_chart).parameters


# --------------------------------------------------------------------------- #
# Data loading & cleaning
# --------------------------------------------------------------------------- #
def find_default_dataset() -> Path | None:
    """Return the first bank-marketing CSV found next to the app."""
    return next((p for p in DATA_CANDIDATES if p.exists()), None)


def _sniff_separator(raw: bytes) -> str:
    header = raw.split(b"\n", 1)[0].decode("utf-8-sig", errors="ignore")
    return ";" if header.count(";") > header.count(",") else ","


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardise columns, handle missing values / duplicates and add derived fields."""
    log: dict = {"raw_rows": len(df), "raw_columns": df.shape[1]}

    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.strip('"').str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "This file does not look like the Bank Marketing dataset. "
            f"Missing columns: {', '.join(missing_cols)}."
        )
    df = df[EXPECTED_COLUMNS]
    log["missing_before"] = int(df.isna().sum().sum())

    # Text columns: trim, lower-case, unknown for blanks
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        df[col] = (df[col].astype("string").str.strip().str.strip('"').str.lower())
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("unknown").astype(str)

    # Numeric columns: coerce, then fill with a sensible neutral value
    fills = {"pdays": -1, "previous": 0}
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(fills.get(col, df[col].median()))

    # Target: rows without a usable yes/no label cannot be analysed
    target_map = {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0}
    df["converted"] = df[TARGET_COL].map(target_map)
    unlabeled = int(df["converted"].isna().sum())
    df = df.dropna(subset=["converted"])
    df["converted"] = df["converted"].astype(int)
    log["unlabeled_dropped"] = unlabeled

    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET_COL] = np.where(df["converted"] == 1, "yes", "no")
    df["converted_label"] = np.where(df["converted"] == 1, "Converted", "Not converted")

    # Derived analysis buckets (ordered categoricals so charts keep a logical order)
    df["age_group"] = pd.cut(df["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf], labels=AGE_LABELS)
    df["campaign_group"] = pd.cut(df["campaign"], [-np.inf, 1, 2, 3, 4, 5, 10, np.inf],
                                  labels=CAMPAIGN_LABELS)
    df["previous_group"] = pd.cut(df["previous"], [-np.inf, 0, 1, 2, 3, 5, np.inf],
                                  labels=PREVIOUS_LABELS)
    df["duration_group"] = pd.cut(df["duration"], [-np.inf, 60, 120, 180, 300, 600, np.inf],
                                  labels=DURATION_LABELS)
    pdays_group = np.select(
        [df["pdays"] < 0, df["pdays"] <= 30, df["pdays"] <= 90, df["pdays"] <= 180,
         df["pdays"] <= 365],
        PDAYS_LABELS[:5], default=PDAYS_LABELS[5])
    df["pdays_group"] = pd.Categorical(pdays_group, categories=PDAYS_LABELS, ordered=True)

    log["final_rows"], log["final_columns"] = len(df), len(EXPECTED_COLUMNS)
    return df, log


@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str | None = None, file_bytes: bytes | None = None) -> tuple[pd.DataFrame, dict]:
    """Read a bank-marketing CSV (semicolon or comma separated) from disk or upload."""
    raw = file_bytes if file_bytes is not None else Path(path).read_bytes()
    df = pd.read_csv(io.BytesIO(raw), sep=_sniff_separator(raw))
    return clean_data(df)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def calculate_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    converted = int(df["converted"].sum())
    return {
        "total": total,
        "converted": converted,
        "non_converted": total - converted,
        "rate": converted / total * 100 if total else 0.0,
        "dropoff_rate": (total - converted) / total * 100 if total else 0.0,
        "avg_campaign": float(df["campaign"].mean()),
        "avg_duration": float(df["duration"].mean()),
    }


def calculate_funnel(df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, dict]:
    """
    Build the campaign funnel from available columns only.

    1. Contacted customers  - every record has campaign >= 1 (observed)
    2. Engaged conversations - call duration >= threshold seconds (DERIVED, user-adjustable)
    3. Subscribed           - engaged calls where y == 'yes' (observed target)
    """
    contacted = int((df["campaign"] >= 1).sum())
    engaged_mask = df["duration"] >= threshold
    engaged = int(engaged_mask.sum())
    subscribed = int((engaged_mask & (df["converted"] == 1)).sum())

    stages = pd.DataFrame({
        "Stage": ["Customers Contacted", f"Engaged Conversations (≥{threshold}s)", "Subscribed"],
        "Customers": [contacted, engaged, subscribed],
        "Basis": ["Observed (campaign ≥ 1)", "Derived (call duration threshold)",
                  "Observed (y = yes) within engaged calls"],
    })
    stages["% of Contacted"] = stages["Customers"] / max(contacted, 1) * 100
    prev = stages["Customers"].shift(1)
    stages["Stage Conversion %"] = np.where(prev.notna(), stages["Customers"] / prev * 100, 100.0)
    stages["Drop-off (count)"] = (prev - stages["Customers"]).fillna(0).astype(int)
    stages["Drop-off %"] = np.where(prev.notna(), 100 - stages["Stage Conversion %"], 0.0)

    total_conv = int(df["converted"].sum())
    extras = {"total_conversions": total_conv, "conversions_outside_engaged": total_conv - subscribed}
    return stages, extras


def rate_table(df: pd.DataFrame, col: str, order: list | None = None,
               sort_desc: bool = False) -> pd.DataFrame:
    """Customers, conversions and conversion rate (%) per category of `col`."""
    g = (df.groupby(col, observed=True)["converted"]
         .agg(customers="count", conversions="sum").reset_index())
    g["rate"] = g["conversions"] / g["customers"] * 100
    if order is not None:
        cats = order + [v for v in g[col].unique() if v not in order]
        g[col] = pd.Categorical(g[col], categories=cats, ordered=True)
        g = g.sort_values(col)
    elif sort_desc:
        g = g.sort_values("rate", ascending=False)
    return g.reset_index(drop=True)


def top_bottom(df: pd.DataFrame, col: str, exclude: tuple = ("unknown",), min_n: int = MIN_N):
    """Best and worst category by conversion rate among segments with enough customers."""
    t = rate_table(df, col)
    t = t[(t["customers"] >= min_n) & (~t[col].astype(str).isin(exclude))]
    if len(t) < 2:
        return None
    return t.loc[t["rate"].idxmax()], t.loc[t["rate"].idxmin()], t


def segment_rate(df: pd.DataFrame, mask: pd.Series) -> tuple[float, int]:
    sub = df[mask]
    return (sub["converted"].mean() * 100 if len(sub) else float("nan")), len(sub)


# --------------------------------------------------------------------------- #
# Insight & recommendation engine (every number comes from the filtered data)
# --------------------------------------------------------------------------- #
def generate_insights(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    n = len(df)
    if n == 0:
        return out
    conv = int(df["converted"].sum())
    overall = conv / n * 100

    def add(category: str, text: str, tag: str = "Association") -> None:
        out.append({"category": category, "text": text, "tag": tag})

    add("Overview", f"{conv:,} of {n:,} contacted customers subscribed in the current selection, "
                    f"an observed conversion rate of {overall:.2f}%.", "Observed Data")

    # Channel
    t = rate_table(df, "contact")
    tb = top_bottom(df, "contact")
    if tb:
        best, worst, _ = tb
        add("Channel", f"Among recorded contact types, '{best['contact']}' shows the highest observed "
                       f"conversion rate ({best['rate']:.1f}%) versus '{worst['contact']}' "
                       f"({worst['rate']:.1f}%).")
    unk = t[t["contact"] == "unknown"]
    if len(unk) and unk["customers"].iloc[0] / n >= 0.05:
        add("Channel", f"The contact type is not recorded for {unk['customers'].iloc[0] / n * 100:.1f}% "
                       f"of customers, and that group converts at {unk['rate'].iloc[0]:.1f}%. "
                       "Channel comparisons should be read with this data gap in mind.", "Observed Data")

    # Timing
    tb = top_bottom(df, "month")
    if tb:
        _, _, mt = tb
        top = mt.sort_values("rate", ascending=False).head(3)
        share = top["customers"].sum() / n * 100
        names = ", ".join(f"{m.capitalize()} ({r:.1f}%)" for m, r in zip(top["month"], top["rate"]))
        add("Timing", f"The highest observed monthly conversion rates are in {names}. "
                      f"These months account for only {share:.1f}% of contacts, so the associations "
                      "rest on smaller volumes.")

    # Previous campaign
    pt = rate_table(df, "poutcome").set_index("poutcome")
    if "success" in pt.index and pt.loc["success", "customers"] >= MIN_N:
        base = pt.loc["unknown"] if "unknown" in pt.index else None
        cmp_txt = (f" versus {base['rate']:.1f}% where no previous outcome is recorded"
                   if base is not None else "")
        add("Campaign History", f"Customers with a successful previous campaign converted at "
                                f"{pt.loc['success', 'rate']:.1f}%{cmp_txt}.")

    # Contact frequency and efficiency
    low_rate, low_n = segment_rate(df, df["campaign"] <= 2)
    high_mask = df["campaign"] >= 6
    high_rate, high_n = segment_rate(df, high_mask)
    if low_n >= MIN_N and high_n >= MIN_N:
        attempts_share = df.loc[high_mask, "campaign"].sum() / df["campaign"].sum() * 100
        conv_share = df.loc[high_mask, "converted"].sum() / max(conv, 1) * 100
        if high_rate < low_rate:
            add("Campaign Frequency",
                f"Higher contact frequency is associated with lower conversion rates: customers "
                f"contacted 6+ times converted at {high_rate:.1f}% versus {low_rate:.1f}% for 1–2 "
                f"contacts. The 6+ group used {attempts_share:.1f}% of all contact attempts but "
                f"produced {conv_share:.1f}% of conversions, suggesting potential diminishing returns.")
        else:
            add("Campaign Frequency",
                f"In this selection, customers contacted 6+ times ({high_rate:.1f}%) did not convert "
                f"less than those contacted 1–2 times ({low_rate:.1f}%).")

    # Call duration (post-call variable)
    long_rate, long_n = segment_rate(df, df["duration"] >= 300)
    short_rate, short_n = segment_rate(df, df["duration"] < 60)
    if long_n >= MIN_N and short_n >= MIN_N:
        add("Call Engagement",
            f"Calls of 5+ minutes converted at {long_rate:.1f}% versus {short_rate:.1f}% for calls under "
            "one minute. Duration is only known after the call, so this describes engagement, not a "
            "lever that can be set in advance.")

    # Customer segments
    for col, label in [("job", "job"), ("education", "education level")]:
        tb = top_bottom(df, col)
        if tb:
            best, worst, _ = tb
            add("Customer Segments", f"By {label}, '{best[col]}' customers show the highest observed "
                                     f"conversion rate ({best['rate']:.1f}%) and '{worst[col]}' the lowest "
                                     f"({worst['rate']:.1f}%).")
    tb = top_bottom(df, "age_group")
    if tb:
        best, worst, _ = tb
        add("Customer Segments", f"Age group {best['age_group']} shows the highest observed conversion "
                                 f"rate ({best['rate']:.1f}%); {worst['age_group']} the lowest "
                                 f"({worst['rate']:.1f}%).")
    h_no, n_no = segment_rate(df, df["housing"] == "no")
    h_yes, n_yes = segment_rate(df, df["housing"] == "yes")
    if n_no >= MIN_N and n_yes >= MIN_N:
        add("Customer Segments", f"Customers without a housing loan convert at {h_no:.1f}% compared with "
                                 f"{h_yes:.1f}% for those with one.")
    if n >= 400 and df["balance"].nunique() > 4:
        q = pd.qcut(df["balance"], 4, duplicates="drop")
        bq = df.groupby(q, observed=True)["converted"].mean() * 100
        if len(bq) >= 2:
            add("Customer Segments", f"The highest account-balance quartile converts at {bq.iloc[-1]:.1f}% "
                                     f"versus {bq.iloc[0]:.1f}% for the lowest quartile.")
    return out


def generate_recommendations(df: pd.DataFrame) -> list[dict]:
    recs: list[dict] = []
    n = len(df)
    if n == 0:
        return recs
    overall = df["converted"].mean() * 100

    def add(title: str, body: str) -> None:
        recs.append({"title": title, "body": body})

    tb = top_bottom(df, "contact")
    if tb:
        best, worst, _ = tb
        add("Prioritise higher-performing contact channels",
            f"Based on the observed data, '{best['contact']}' contacts convert at {best['rate']:.1f}% "
            f"versus {worst['rate']:.1f}% for '{worst['contact']}'. This association suggests shifting "
            f"outreach effort toward '{best['contact']}' where it is operationally feasible.")

    low_rate, low_n = segment_rate(df, df["campaign"] <= 2)
    high_rate, high_n = segment_rate(df, df["campaign"] >= 6)
    if low_n >= MIN_N and high_n >= MIN_N and high_rate < low_rate:
        add("Review contact-frequency limits",
            f"Based on the observed data, customers contacted 6+ times convert at {high_rate:.1f}% "
            f"compared with {low_rate:.1f}% for 1–2 contacts. This association suggests testing a "
            "contact cap (for example an A/B test) before assuming repeated calls are cost-effective.")

    pt = rate_table(df, "poutcome").set_index("poutcome")
    if "success" in pt.index and pt.loc["success", "customers"] >= MIN_N:
        s = pt.loc["success"]
        add("Re-engage customers with a successful prior campaign",
            f"Based on the observed data, {int(s['customers']):,} customers had a successful previous "
            f"outcome and converted at {s['rate']:.1f}% (overall: {overall:.1f}%). This association "
            "suggests these customers are a high-priority re-engagement audience.")

    parts = []
    for col in ["job", "age_group"]:
        r = top_bottom(df, col)
        if r:
            parts.append(f"{col.replace('_', ' ')} '{r[0][col]}' ({r[0]['rate']:.1f}%)")
    if parts:
        add("Segment campaigns by customer characteristics",
            f"Based on the observed data, the highest-converting groups include {' and '.join(parts)}. "
            "This association suggests tailoring messaging and offers per segment; segment size and "
            "acquisition cost should be checked before scaling.")

    mt = top_bottom(df, "month")
    if mt:
        _, _, tbl = mt
        top = tbl.sort_values("rate", ascending=False).head(3)
        names = ", ".join(m.capitalize() for m in top["month"])
        share = top["customers"].sum() / n * 100
        add("Optimise campaign timing",
            f"Based on the observed data, {names} show the highest conversion rates but receive only "
            f"{share:.1f}% of contacts. This association suggests piloting additional activity in those "
            "months while checking whether the pattern reflects seasonality or customer mix.")

    unk = rate_table(df, "contact")
    unk = unk[unk["contact"] == "unknown"]
    if len(unk) and unk["customers"].iloc[0] / n >= 0.05:
        add("Improve contact-channel data capture",
            f"Based on the observed data, {unk['customers'].iloc[0] / n * 100:.1f}% of records have no "
            "contact type. Recording the channel consistently would make channel-level decisions more "
            "reliable.")

    add("Validate with experiments",
        "All findings here are associations in historical data. Controlled tests on channel, timing "
        "and frequency are needed before treating any of them as causal effects.")
    return recs


def build_report(df: pd.DataFrame, kpis: dict) -> str:
    """Plain-text summary report for download."""
    lines = ["MARKETING FUNNEL ANALYTICS - SUMMARY REPORT",
             "Future Interns - Data Science & Analytics Task 3", "=" * 60, "",
             "OBSERVED DATA",
             f"Total customers contacted : {kpis['total']:,}",
             f"Conversions (y = yes)     : {kpis['converted']:,}",
             f"Conversion rate           : {kpis['rate']:.2f}%",
             f"Average campaign contacts : {kpis['avg_campaign']:.2f}",
             f"Average call duration     : {kpis['avg_duration']:.0f} seconds", ""]
    lines.append("BEST-PERFORMING SEGMENTS (segments with at least "
                 f"{MIN_N} customers, 'unknown' excluded)")
    for col, label in [("contact", "Channel"), ("job", "Job"), ("education", "Education"),
                       ("age_group", "Age group"), ("month", "Month")]:
        tb = top_bottom(df, col)
        if tb:
            lines.append(f"{label:<10}: {tb[0][col]} ({tb[0]['rate']:.1f}% conversion)")
    lines += ["", "INSIGHTS (associations, not causation)"]
    lines += [f"- [{i['category']}] {i['text']}" for i in generate_insights(df)]
    lines += ["", "RECOMMENDATIONS"]
    lines += [f"{k}. {r['title']}: {r['body']}" for k, r in enumerate(generate_recommendations(df), 1)]
    lines += ["", "Dataset: Moro, Laureano & Cortez (2011), Bank Marketing (UCI)."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Predictive model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Training model…")
def train_model(data: pd.DataFrame, model_name: str, features: tuple, seed: int = 42) -> dict:
    """Train/test a conversion model on the given feature set (no target leakage)."""
    cat_cols = [c for c in features if c in CATEGORICAL_COLS]
    num_cols = [c for c in features if c not in CATEGORICAL_COLS]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if model_name == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5,
                                           class_weight="balanced_subsample", n_jobs=-1,
                                           random_state=seed)
    else:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("prep", ColumnTransformer(transformers)), ("model", estimator)])

    X, y = data[list(features)], data["converted"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Aggregate encoded feature importance back to original columns
    names = pipe.named_steps["prep"].get_feature_names_out()
    raw = (estimator.feature_importances_ if model_name == "Random Forest"
           else np.abs(estimator.coef_[0]))
    origin = []
    for nm in names:
        base = nm.split("__", 1)[1]
        match = [c for c in features if base == c or base.startswith(c + "_")]
        origin.append(max(match, key=len) if match else base)
    imp = (pd.DataFrame({"feature": origin, "importance": raw}).groupby("feature")["importance"]
           .sum().sort_values(ascending=False).reset_index())
    imp["importance"] = imp["importance"] / imp["importance"].sum() * 100

    # Lift by decile of predicted probability
    ranks = pd.Series(proba).rank(method="first", ascending=False)
    decile = pd.qcut(ranks, 10, labels=range(1, 11))
    dec = (pd.DataFrame({"decile": decile, "actual": y_te.to_numpy()})
           .groupby("decile", observed=True)["actual"].mean().mul(100).reset_index(name="rate"))
    dec["lift"] = dec["rate"] / (y_te.mean() * 100)

    return {
        "metrics": {
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "F1-score": f1_score(y_te, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_te, proba),
        },
        "majority_baseline": 1 - y_te.mean(),
        "confusion": confusion_matrix(y_te, pred, labels=[0, 1]),
        "importance": imp, "deciles": dec,
        "n_train": len(X_tr), "n_test": len(X_te),
    }


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def show(fig: go.Figure) -> None:
    """Render a Plotly figure across Streamlit versions."""
    if _PLOTLY_NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def style_fig(fig: go.Figure, title: str, x_title: str | None = None,
              y_title: str | None = None, height: int = 400) -> go.Figure:
    grid = "rgba(128,128,128,0.18)"
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16)),
        height=height, margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(title=x_title, gridcolor=grid, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=grid, zeroline=False)
    return fig


def create_funnel(stages: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=stages["Stage"], x=stages["Customers"],
        textinfo="value+percent initial",
        marker=dict(color=[C_PRIMARY, "#6366F1", C_POS][: len(stages)]),
        connector=dict(line=dict(color="rgba(128,128,128,0.3)")),
        customdata=stages[["Stage Conversion %", "Basis"]],
        hovertemplate="<b>%{y}</b><br>Customers: %{x:,}<br>Stage conversion: "
                      "%{customdata[0]:.1f}%<br>Basis: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(funnelmode="stack")
    return style_fig(fig, "Campaign Funnel", height=360)


def create_conversion_chart(tbl: pd.DataFrame, x: str, title: str, x_label: str,
                            overall_rate: float, horizontal: bool = False,
                            height: int = 400) -> go.Figure:
    """Conversion-rate bar chart with volume in the hover and an overall-rate reference line."""
    d = tbl.copy()
    d[x] = d[x].astype(str)
    cd = d[["customers", "conversions"]].to_numpy()
    text = d["rate"].map(lambda v: f"{v:.1f}%")
    if horizontal:
        fig = go.Figure(go.Bar(
            y=d[x], x=d["rate"], orientation="h", text=text, textposition="outside",
            marker_color=C_PRIMARY, customdata=cd,
            hovertemplate="<b>%{y}</b><br>Conversion rate: %{x:.2f}%<br>Customers: %{customdata[0]:,}"
                          "<br>Conversions: %{customdata[1]:,}<extra></extra>"))
        fig.add_vline(x=overall_rate, line_dash="dash", line_color=C_WARN,
                      annotation_text=f"Overall {overall_rate:.1f}%", annotation_position="top")
        fig.update_yaxes(autorange="reversed")
        style_fig(fig, title, "Conversion rate (%)", x_label, height)
    else:
        fig = go.Figure(go.Bar(
            x=d[x], y=d["rate"], text=text, textposition="outside", marker_color=C_PRIMARY,
            customdata=cd,
            hovertemplate="<b>%{x}</b><br>Conversion rate: %{y:.2f}%<br>Customers: %{customdata[0]:,}"
                          "<br>Conversions: %{customdata[1]:,}<extra></extra>"))
        fig.add_hline(y=overall_rate, line_dash="dash", line_color=C_WARN,
                      annotation_text=f"Overall {overall_rate:.1f}%", annotation_position="top left")
        style_fig(fig, title, x_label, "Conversion rate (%)", height)
    fig.update_traces(cliponaxis=False)
    return fig


def create_rate_volume_chart(tbl: pd.DataFrame, x: str, title: str, x_label: str,
                             overall_rate: float, height: int = 400) -> go.Figure:
    """Customer volume (bars) with conversion-rate line on a secondary axis."""
    d = tbl.copy()
    d[x] = d[x].astype(str)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d[x], y=d["customers"], name="Customers", marker_color="rgba(148,163,184,0.45)",
                hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<extra></extra>", secondary_y=False)
    fig.add_scatter(x=d[x], y=d["rate"], name="Conversion rate", mode="lines+markers",
                    line=dict(color=C_PRIMARY, width=3), marker=dict(size=8),
                    customdata=d[["conversions"]].to_numpy(),
                    hovertemplate="<b>%{x}</b><br>Conversion rate: %{y:.2f}%<br>Conversions: "
                                  "%{customdata[0]:,}<extra></extra>", secondary_y=True)
    fig.add_hline(y=overall_rate, line_dash="dash", line_color=C_WARN, secondary_y=True,
                  annotation_text=f"Overall {overall_rate:.1f}%", annotation_position="top left")
    style_fig(fig, title, x_label, None, height)
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="Conversion rate (%)", secondary_y=True, showgrid=False)
    return fig


def create_donut(kpis: dict) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Converted", "Not converted"], values=[kpis["converted"], kpis["non_converted"]],
        hole=0.62, marker=dict(colors=[C_POS, C_NEG]), textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{kpis['rate']:.1f}%</b><br>converted", showarrow=False,
                       font=dict(size=18))
    style_fig(fig, "Conversion vs Non-Conversion", height=360)
    return fig


def create_heatmap(df: pd.DataFrame, rows: str, cols: str, title: str, order: list | None) -> go.Figure:
    pivot = df.pivot_table(index=rows, columns=cols, values="converted", aggfunc="mean", observed=True)
    counts = df.pivot_table(index=rows, columns=cols, values="converted", aggfunc="count", observed=True)
    pivot = (pivot * 100).where(counts >= 30)  # hide cells with fewer than 30 customers
    if order:
        pivot = pivot.reindex([o for o in order if o in pivot.index])
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", text_auto=".1f",
                    labels=dict(color="Conversion rate (%)", x=cols.capitalize(), y=rows.capitalize()))
    fig.update_traces(hovertemplate=f"{rows.capitalize()}: %{{y}}<br>{cols.capitalize()}: %{{x}}"
                                    "<br>Conversion rate: %{z:.2f}%<extra></extra>")
    return style_fig(fig, title, height=420)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1400px}
#MainMenu,footer{visibility:hidden}
.hero{padding:1.7rem 2.1rem;border-radius:20px;color:#fff;margin-bottom:1.3rem;
background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 58%,#0e7490 100%)}
.hero .eyebrow{display:inline-block;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;
padding:.25rem .7rem;border-radius:999px;background:rgba(255,255,255,.14);margin-bottom:.7rem}
.hero h1{margin:0;font-size:1.9rem;letter-spacing:.04em;color:#fff;padding:0}
.hero .tag{font-size:1.05rem;opacity:.92;margin:.25rem 0 .55rem}
.hero .sub{font-size:.9rem;opacity:.75;max-width:760px}
.kpi{border:1px solid rgba(128,128,128,.28);background:rgba(128,128,128,.07);border-radius:16px;
padding:1rem 1.15rem;height:100%}
.kpi .l{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;opacity:.72}
.kpi .v{font-size:1.85rem;font-weight:700;line-height:1.25;margin:.15rem 0}
.kpi .s{font-size:.75rem;opacity:.6}
.sec{margin:1.6rem 0 .6rem}
.sec h3{margin:0;padding:0;font-size:1.25rem}
.sec p{margin:.15rem 0 0;font-size:.88rem;opacity:.68}
.card{border:1px solid rgba(128,128,128,.25);border-left:4px solid #3B82F6;
background:rgba(128,128,128,.06);border-radius:12px;padding:.85rem 1.1rem;margin-bottom:.7rem}
.card .h{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;opacity:.7;margin-bottom:.2rem}
.card .b{font-size:.95rem;line-height:1.5}
.card.rec{border-left-color:#14B8A6}
.pill{display:inline-block;font-size:.66rem;padding:.08rem .5rem;border-radius:999px;margin-left:.5rem;
border:1px solid rgba(128,128,128,.4);opacity:.85;text-transform:none;letter-spacing:0}
</style>""", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        '<div class="hero"><span class="eyebrow">Future Interns • Data Science &amp; Analytics Task 3</span>'
        "<h1>MARKETING FUNNEL ANALYTICS</h1>"
        '<div class="tag">Campaign Performance &amp; Conversion Intelligence</div>'
        '<div class="sub">Interactive analysis of customer campaign performance, conversion behavior '
        "and marketing efficiency.</div></div>", unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="sec"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def render_kpis(k: dict) -> None:
    mins, secs = divmod(int(round(k["avg_duration"])), 60)
    cards = [
        ("👥 Total Customers", f"{k['total']:,}", "Observed: contacted records"),
        ("🎯 Converted", f"{k['converted']:,}", "Observed: y = yes"),
        ("📈 Conversion Rate", f"{k['rate']:.2f}%", "Derived: converted ÷ total"),
        ("❌ Non-Converted", f"{k['non_converted']:,}", f"Drop-off rate {k['dropoff_rate']:.1f}%"),
        ("📞 Avg Campaign Contacts", f"{k['avg_campaign']:.2f}", "Contacts per customer"),
        ("⏱ Avg Call Duration", f"{mins}m {secs:02d}s", f"{k['avg_duration']:.0f} seconds"),
    ]
    for col, (label, value, sub) in zip(st.columns(6), cards):
        col.markdown(kpi_card(label, value, sub), unsafe_allow_html=True)


def render_cards(items: list[dict], kind: str = "insight") -> None:
    for i, it in enumerate(items, 1):
        if kind == "insight":
            head = f'{it["category"]}<span class="pill">{it["tag"]}</span>'
            body, cls = it["text"], "card"
        else:
            head = f'{i}. {it["title"]}<span class="pill">Recommendation</span>'
            body, cls = it["body"], "card rec"
        st.markdown(f'<div class="{cls}"><div class="h">{head}</div><div class="b">{body}</div></div>',
                    unsafe_allow_html=True)


def two_cols(fig_left: go.Figure, fig_right: go.Figure) -> None:
    left, right = st.columns(2)
    with left:
        show(fig_left)
    with right:
        show(fig_right)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """Apply sidebar selections. Empty multiselects mean 'include all'."""
    mask = pd.Series(True, index=df.index)
    for col, values in sel["multi"].items():
        if values:
            mask &= df[col].isin(values)
    for col, (lo, hi) in sel["ranges"].items():
        mask &= df[col].between(lo, hi)
    return df[mask]


def render_filters(df: pd.DataFrame) -> dict:
    ver = st.session_state.setdefault("filter_version", 0)
    k = lambda name: f"{name}_{ver}"  # changing the version resets every widget

    multi_spec = [("job", "Job"), ("marital", "Marital status"), ("education", "Education"),
                  ("housing", "Housing loan"), ("loan", "Personal loan"), ("contact", "Contact type"),
                  ("month", "Month"), ("poutcome", "Previous outcome")]
    multi: dict = {}
    with st.sidebar.expander("Filters", expanded=False):
        for col, label in multi_spec:
            options = sorted(df[col].unique(), key=MONTH_ORDER.index if col == "month" else str)
            multi[col] = st.multiselect(label, options, key=k(col), placeholder="All")
        ranges: dict = {}
        for col, label in [("age", "Age range"), ("balance", "Balance range (€)"),
                           ("campaign", "Campaign contacts")]:
            lo, hi = int(df[col].min()), int(df[col].max())
            ranges[col] = st.slider(label, lo, hi, (lo, hi), key=k(col)) if lo < hi else (lo, hi)
        st.button("Reset Filters", use_container_width=True,
                  on_click=lambda: st.session_state.update(filter_version=ver + 1))
    return {"multi": multi, "ranges": ranges}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_executive(df, kpis, threshold):
    render_kpis(kpis)
    section("Funnel & outcome", "Observed conversion outcome and the derived campaign funnel.")
    stages, _ = calculate_funnel(df, threshold)
    left, right = st.columns([3, 2])
    with left:
        show(create_funnel(stages))
    with right:
        show(create_donut(kpis))
    section("Top insights", "Calculated live from the current filter selection.")
    render_cards(generate_insights(df)[:5])
    section("Business recommendations", "Suggested actions derived from the associations above.")
    render_cards(generate_recommendations(df)[:3], kind="rec")


def page_funnel(df, kpis, threshold):
    st.info("**Methodology.** The funnel is derived from the available campaign dataset and does not "
            "represent website traffic unless such data exists. The dataset has no visitor or lead "
            "stages, so none are shown. Every record is a contacted customer (`campaign ≥ 1`). The "
            f"**engaged conversation** stage is *derived*: calls lasting at least **{threshold} seconds** "
            "(adjust in the sidebar). The final stage uses the observed target `y = yes`.")
    stages, extras = calculate_funnel(df, threshold)
    left, right = st.columns([3, 2])
    with left:
        show(create_funnel(stages))
    with right:
        prog = go.Figure(go.Scatter(
            x=stages["Stage"], y=stages["% of Contacted"], mode="lines+markers+text",
            text=stages["% of Contacted"].map(lambda v: f"{v:.1f}%"), textposition="top center",
            line=dict(color=C_PRIMARY, width=3), marker=dict(size=10),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}% of contacted<extra></extra>"))
        prog.update_yaxes(range=[0, 115])
        show(style_fig(prog, "Conversion Progression", "Funnel stage", "% of contacted customers", 360))

    section("Stage metrics", "Stage conversion = customers at this stage ÷ customers at the previous stage.")
    table = stages.copy()
    st.dataframe(table.style.format({
        "Customers": "{:,}", "% of Contacted": "{:.2f}%", "Stage Conversion %": "{:.2f}%",
        "Drop-off (count)": "{:,}", "Drop-off %": "{:.2f}%"}), hide_index=True)
    st.caption(f"Reconciliation: {extras['total_conversions']:,} total conversions, of which "
               f"{extras['conversions_outside_engaged']:,} occurred in calls shorter than {threshold}s "
               "and are therefore not in the final funnel stage. Overall conversion "
               f"(y = yes ÷ all customers) is {kpis['rate']:.2f}%.")

    section("Sensitivity to the engagement threshold",
            "Because the engaged stage is an assumption, this shows how it changes with the threshold.")
    rows = []
    for t in [0, 30, 60, 120, 180, 300, 600]:
        m = df["duration"] >= t
        eng = int(m.sum())
        rows.append({"Threshold (s)": t, "Engaged share": eng / len(df) * 100,
                     "Conversion among engaged": df.loc[m, "converted"].mean() * 100 if eng else np.nan,
                     "Conversions retained": df.loc[m, "converted"].sum() / max(kpis["converted"], 1) * 100})
    sens = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color in [("Engaged share", C_NEG), ("Conversion among engaged", C_PRIMARY),
                       ("Conversions retained", C_POS)]:
        fig.add_scatter(x=sens["Threshold (s)"], y=sens[col], name=col, mode="lines+markers",
                        line=dict(color=color, width=3),
                        hovertemplate=f"{col}: %{{y:.2f}}%<br>Threshold: %{{x}}s<extra></extra>")
    show(style_fig(fig, "Funnel Sensitivity to Call-Duration Threshold", "Engagement threshold (seconds)",
                   "Percent (%)", 380))


def page_channel(df, kpis, threshold):
    rate = kpis["rate"]
    st.caption("Derived metrics: conversion rate = conversions ÷ customers in each group. Dashed line = overall rate.")
    two_cols(
        create_conversion_chart(rate_table(df, "contact", sort_desc=True), "contact",
                                "Conversion Rate by Contact Type", "Contact type", rate),
        create_conversion_chart(rate_table(df, "poutcome", sort_desc=True), "poutcome",
                                "Conversion Rate by Previous Campaign Outcome", "Previous outcome", rate))
    two_cols(
        create_rate_volume_chart(rate_table(df, "month", order=MONTH_ORDER), "month",
                                 "Conversion Rate by Month", "Month", rate),
        create_rate_volume_chart(rate_table(df, "campaign_group", order=CAMPAIGN_LABELS), "campaign_group",
                                 "Conversion Rate by Campaign Contact Frequency", "Contacts in this campaign", rate))
    two_cols(
        create_conversion_chart(rate_table(df, "previous_group", order=PREVIOUS_LABELS), "previous_group",
                                "Conversion Rate by Number of Previous Contacts", "Previous contacts", rate),
        create_heatmap(df, "month", "contact", "Conversion Rate: Month × Contact Type (cells ≥ 30 customers)",
                       MONTH_ORDER))
    st.caption("Low-volume months and groups can show extreme rates; check the customer counts in the hover text.")


def page_segments(df, kpis, threshold):
    rate = kpis["rate"]
    two_cols(
        create_conversion_chart(rate_table(df, "job", sort_desc=True), "job",
                                "Conversion Rate by Job", "Job", rate, horizontal=True, height=470),
        create_conversion_chart(rate_table(df, "age_group", order=AGE_LABELS), "age_group",
                                "Conversion Rate by Age Group", "Age group", rate, height=470))
    two_cols(
        create_conversion_chart(rate_table(df, "education", sort_desc=True), "education",
                                "Conversion Rate by Education", "Education", rate),
        create_conversion_chart(rate_table(df, "marital", sort_desc=True), "marital",
                                "Conversion Rate by Marital Status", "Marital status", rate))
    section("Financial profile", "Housing loan, personal loan and credit default status.")
    cols = st.columns(3)
    for c, (col, label) in zip(cols, [("housing", "Housing Loan"), ("loan", "Personal Loan"),
                                      ("default", "Credit Default")]):
        with c:
            show(create_conversion_chart(rate_table(df, col, sort_desc=True), col,
                                         f"Conversion by {label}", label, rate, height=340))
    section("Distributions", "Observed age and balance distributions by outcome.")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="age", color="converted_label", nbins=40, barmode="overlay", opacity=0.7,
                           color_discrete_map={"Converted": C_POS, "Not converted": C_NEG})
        fig.update_traces(hovertemplate="Age: %{x}<br>Customers: %{y:,}<extra></extra>")
        show(style_fig(fig, "Age Distribution by Outcome", "Age", "Customers", 380))
    with right:
        fig = px.box(df, x="converted_label", y="balance", color="converted_label", points=False,
                     color_discrete_map={"Converted": C_POS, "Not converted": C_NEG})
        lo, hi = df["balance"].quantile([0.02, 0.98])
        fig.update_yaxes(range=[lo, hi])
        fig.update_layout(showlegend=False)
        show(style_fig(fig, "Account Balance by Outcome (2nd–98th percentile view)", "Outcome",
                       "Balance (€)", 380))


def page_performance(df, kpis, threshold):
    rate = kpis["rate"]
    two_cols(
        create_rate_volume_chart(rate_table(df, "campaign_group", order=CAMPAIGN_LABELS), "campaign_group",
                                 "Conversion Rate vs Number of Contacts", "Contacts in this campaign", rate),
        create_rate_volume_chart(rate_table(df, "duration_group", order=DURATION_LABELS), "duration_group",
                                 "Conversion Rate vs Call Duration", "Call duration", rate))
    st.caption("Call duration is only known after a call ends, so it describes engagement rather than "
               "a variable that can be planned in advance. Higher duration is associated with higher "
               "conversion, not proven to cause it.")
    two_cols(
        create_conversion_chart(rate_table(df, "poutcome", sort_desc=True), "poutcome",
                                "Previous Outcome vs Current Conversion", "Previous outcome", rate),
        create_conversion_chart(rate_table(df, "pdays_group", order=PDAYS_LABELS), "pdays_group",
                                "Conversion Rate by Days Since Previous Campaign Contact (pdays)",
                                "Days since previous contact", rate, horizontal=True))

    section("Effort vs return: diminishing returns check",
            "Share of all contact attempts versus share of all conversions, by contacts per customer.")
    g = df.groupby("campaign_group", observed=True).agg(attempts=("campaign", "sum"),
                                                        conversions=("converted", "sum")).reset_index()
    g["Share of contact attempts"] = g["attempts"] / max(g["attempts"].sum(), 1) * 100
    g["Share of conversions"] = g["conversions"] / max(g["conversions"].sum(), 1) * 100
    g["campaign_group"] = g["campaign_group"].astype(str)
    fig = go.Figure()
    fig.add_bar(x=g["campaign_group"], y=g["Share of contact attempts"], name="Share of contact attempts",
                marker_color=C_WARN, hovertemplate="%{x} contacts<br>%{y:.1f}% of attempts<extra></extra>")
    fig.add_bar(x=g["campaign_group"], y=g["Share of conversions"], name="Share of conversions",
                marker_color=C_POS, hovertemplate="%{x} contacts<br>%{y:.1f}% of conversions<extra></extra>")
    fig.update_layout(barmode="group")
    show(style_fig(fig, "Contact Effort vs Conversion Contribution", "Contacts per customer",
                   "Share of total (%)", 400))

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df[df["duration"] <= df["duration"].quantile(0.99)], x="duration",
                           color="converted_label", nbins=60, barmode="overlay", opacity=0.7,
                           color_discrete_map={"Converted": C_POS, "Not converted": C_NEG})
        fig.update_traces(hovertemplate="Duration: %{x}s<br>Customers: %{y:,}<extra></extra>")
        show(style_fig(fig, "Call Duration Distribution (up to 99th percentile)", "Duration (seconds)",
                       "Customers", 380))
    with right:
        cap = df[df["campaign"] <= 15]
        fig = px.box(cap, x="converted_label", y="campaign", color="converted_label", points=False,
                     color_discrete_map={"Converted": C_POS, "Not converted": C_NEG})
        fig.update_layout(showlegend=False)
        show(style_fig(fig, "Campaign Contacts by Outcome (up to 15 contacts)", "Outcome",
                       "Contacts in this campaign", 380))


def page_insights(df, kpis, threshold):
    st.caption("Insights are generated from the current filter selection. Segments smaller than "
               f"{MIN_N} customers and 'unknown' categories are excluded from best/worst comparisons.")
    insights = generate_insights(df)
    groups = {}
    for it in insights:
        groups.setdefault(it["category"], []).append(it)
    for category, items in groups.items():
        section(category)
        render_cards(items)


def page_recommendations(df, kpis, threshold):
    st.warning("Recommendations are based on associations observed in historical data. They are "
               "hypotheses to test, not proof that a change will cause higher conversion.")
    render_cards(generate_recommendations(df), kind="rec")


def page_predictive(df, kpis, threshold):
    section("Campaign Outcome Prediction",
            "Estimates which customers are likely to subscribe using information available before a call.")
    st.info("**Leakage note.** `duration` (call length) is only known after the call, and it is strongly "
            "linked to the outcome. Using it would make the model look accurate but useless for choosing "
            "whom to call, so it is **excluded by default**. `campaign` counts contacts including the "
            "current one, so treat it as contact-history information.")

    c1, c2 = st.columns([1, 2])
    model_name = c1.selectbox("Model", ["Logistic Regression", "Random Forest"])
    groups = c2.multiselect("Feature groups", list(FEATURE_GROUPS), default=list(FEATURE_GROUPS))
    include_duration = st.checkbox("Include call duration (post-call variable, causes leakage)", value=False)
    features = [f for g in groups for f in FEATURE_GROUPS[g]] + (["duration"] if include_duration else [])
    if not features:
        st.warning("Select at least one feature group.")
        return
    pos = int(df["converted"].sum())
    if len(df) < 500 or pos < 30 or len(df) - pos < 30:
        st.warning("The current filter leaves too few customers (or too few of one outcome) to train a "
                   "reliable model. Widen the filters (need 500+ rows and 30+ of each outcome).")
        return

    res = train_model(df[features + ["converted"]], model_name, tuple(features))
    st.caption(f"Trained on the filtered data: {res['n_train']:,} training rows, {res['n_test']:,} "
               "held-out test rows (stratified 75/25 split). Class weights are balanced.")
    for col, (name, val) in zip(st.columns(5), res["metrics"].items()):
        col.markdown(kpi_card(name, f"{val:.3f}"), unsafe_allow_html=True)
    st.caption(f"Accuracy alone is misleading here: always predicting 'no' would score "
               f"{res['majority_baseline']:.3f}. Precision, recall and ROC AUC are more informative.")
    if include_duration:
        st.error("Call duration is included. These metrics are not achievable before making a call.")

    left, right = st.columns(2)
    with left:
        fig = px.imshow(res["confusion"], text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
                        labels=dict(color="Customers"))
        fig.update_traces(hovertemplate="%{y}, %{x}<br>Customers: %{z:,}<extra></extra>")
        show(style_fig(fig, "Confusion Matrix (test set)", "Predicted class", "Actual class", 400))
    with right:
        imp = res["importance"].head(12)
        label = "Random-forest importance" if model_name == "Random Forest" else "Share of |coefficient| weight"
        fig = go.Figure(go.Bar(y=imp["feature"], x=imp["importance"], orientation="h", marker_color=C_PRIMARY,
                               hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style_fig(fig, "Feature Importance (aggregated by original column)", f"{label} (%)", "Feature", 400))

    dec = res["deciles"]
    fig = go.Figure(go.Bar(x=dec["decile"].astype(str), y=dec["rate"], marker_color=C_POS,
                           customdata=dec[["lift"]].to_numpy(),
                           hovertemplate="Decile %{x}<br>Actual conversion: %{y:.1f}%<br>Lift: %{customdata[0]:.2f}×<extra></extra>"))
    fig.add_hline(y=kpis["rate"], line_dash="dash", line_color=C_WARN,
                  annotation_text=f"Overall {kpis['rate']:.1f}%", annotation_position="top right")
    show(style_fig(fig, "Actual Conversion by Predicted-Probability Decile (1 = highest scored)",
                   "Score decile", "Actual conversion rate (%)", 380))
    st.caption("Feature importance and model scores describe statistical association in this dataset, "
               "not causal drivers of subscription.")


def page_quality(df, raw_log, kpis, threshold):
    with st.expander("Data Quality Report", expanded=True):
        a, b, c, d = st.columns(4)
        a.metric("Rows (filtered)", f"{len(df):,}")
        b.metric("Rows (full dataset)", f"{raw_log['final_rows']:,}")
        c.metric("Columns", raw_log["final_columns"])
        d.metric("Duplicate rows removed", raw_log["duplicates_removed"])
        e, f, g, h = st.columns(4)
        e.metric("Missing values (original)", raw_log["missing_before"])
        f.metric("Rows without target dropped", raw_log["unlabeled_dropped"])
        g.metric("Converted (y = yes)", f"{kpis['converted']:,}")
        h.metric("Not converted (y = no)", f"{kpis['non_converted']:,}")

        cols = EXPECTED_COLUMNS
        profile = pd.DataFrame({
            "Column": cols,
            "Data type": [str(df[c].dtype) for c in cols],
            "Missing": [int(df[c].isna().sum()) for c in cols],
            "Unique values": [int(df[c].nunique()) for c in cols],
        })
        st.dataframe(profile, hide_index=True)
        dist = df[TARGET_COL].value_counts().rename_axis("Target (y)").reset_index(name="Customers")
        dist["Share"] = dist["Customers"] / dist["Customers"].sum() * 100
        st.dataframe(dist.style.format({"Customers": "{:,}", "Share": "{:.2f}%"}), hide_index=True)
        st.caption("The target is imbalanced: most customers did not subscribe. Cleaning steps: column names "
                   "standardised, text trimmed/lower-cased, blanks set to 'unknown', numeric blanks filled "
                   "(pdays = -1 means not previously contacted), exact duplicates removed.")
    with st.expander("Numeric summary"):
        st.dataframe(df[NUMERIC_COLS].describe().T.style.format("{:,.2f}"))
    with st.expander("Data preview (first 200 rows)"):
        st.dataframe(df[EXPECTED_COLUMNS].head(200), hide_index=True)


PAGE_FUNCS = {
    "Executive Dashboard": page_executive,
    "Funnel Analysis": page_funnel,
    "Channel & Campaign": page_channel,
    "Customer Segmentation": page_segments,
    "Campaign Performance": page_performance,
    "Customer Insights": page_insights,
    "Recommendations": page_recommendations,
    "Predictive Insights": page_predictive,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Marketing Funnel Analytics", page_icon="📊", layout="wide")
    inject_css()

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=64)
        st.markdown("### Marketing Funnel Analytics")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        upload = st.file_uploader("Upload a Bank Marketing CSV", type=["csv"],
                                  help="Optional. Overrides the bundled bank-full.csv.")

    # --- load data -------------------------------------------------------- #
    try:
        if upload is not None:
            full_df, log = load_data(file_bytes=upload.getvalue())
            source = f"Uploaded: {upload.name}"
        else:
            default = find_default_dataset()
            if default is None:
                render_header()
                st.error("No dataset found. Place `bank-full.csv` in the project folder (or in `data/`), "
                         "or upload a CSV from the sidebar.")
                st.stop()
            full_df, log = load_data(path=str(default))
            source = f"Bundled: {default.name}"
    except ValueError as exc:
        render_header()
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # unreadable / malformed file
        render_header()
        st.error(f"The file could not be read as a CSV: {exc}")
        st.stop()

    # --- sidebar controls ------------------------------------------------- #
    st.sidebar.caption(f"{source} • {log['final_rows']:,} rows × {log['final_columns']} columns")
    selection = render_filters(full_df)
    with st.sidebar.expander("Funnel assumption"):
        threshold = st.slider("Engaged-conversation threshold (seconds)", 0, 600, 60, 10,
                              help="Calls at least this long count as an 'engaged conversation' "
                                   "in the derived funnel stage.")
    df = apply_filters(full_df, selection)

    render_header()
    if df.empty:
        st.warning("No customers match the current filters. Use **Reset Filters** in the sidebar.")
        st.stop()
    kpis = calculate_kpis(df)
    if len(df) != len(full_df):
        st.caption(f"Filters active: showing {len(df):,} of {len(full_df):,} customers.")

    # --- exports ----------------------------------------------------------- #
    st.sidebar.divider()
    st.sidebar.download_button("Download filtered CSV", df[EXPECTED_COLUMNS].to_csv(index=False).encode("utf-8"),
                               "filtered_bank_marketing.csv", "text/csv")
    st.sidebar.download_button("Download summary report", build_report(df, kpis).encode("utf-8"),
                               "marketing_summary_report.txt", "text/plain")

    # --- route -------------------------------------------------------------- #
    if page == "Data Quality & Export":
        page_quality(df, log, kpis, threshold)
    else:
        PAGE_FUNCS[page](df, kpis, threshold)
    st.caption("Dataset: Moro, Laureano & Cortez (2011), Bank Marketing, UCI Machine Learning Repository.")


if __name__ == "__main__":
    main()
