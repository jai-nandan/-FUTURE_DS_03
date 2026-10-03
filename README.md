# Marketing Funnel & Conversion Performance Analysis

**Future Interns — Data Science & Analytics, Task 3 (2026)**

An interactive Streamlit dashboard that turns the UCI **Bank Marketing** dataset into a
growth-analytics product: campaign funnel, conversion KPIs, channel and segment
performance, a leakage-aware predictive model, and data-driven recommendations.

---

## 1. Project Overview

A Portuguese bank ran phone-based term-deposit campaigns between 2008 and 2010. This
project analyzes those campaigns end-to-end — from "who was contacted" to "who
subscribed" — and surfaces where conversion is strongest, where it drops off, and what
the bank could test next.

## 2. Business Problem

Marketing and call-center teams need to know:

- How many customers were contacted, and how many converted?
- Which channels, months, and contact frequencies perform best?
- Which customer segments are most responsive?
- Can next-call conversion likelihood be estimated *before* the call happens?
- What should the team change next, based on evidence rather than guesswork?

## 3. Dataset

- **Source:** [Moro, S., Laureano, R., & Cortez, P. (2011)](http://hdl.handle.net/1822/14838),
  *Using Data Mining for Bank Direct Marketing*, UCI Machine Learning Repository.
- **Files:** `bank-full.csv` (45,211 rows) and `bank.csv` (4,521 rows, a 10% sample) —
  both ship with this app and both are auto-detected.
- **Target:** `y` — did the customer subscribe to a term deposit (`yes`/`no`)?
- No missing values in the raw files; the app still runs a full cleaning and
  validation pass and will not silently misread a malformed upload.

## 4. Objectives

1. Quantify the campaign funnel and overall conversion rate.
2. Compare performance across contact channels, timing, and contact frequency.
3. Compare performance across customer segments (job, age, education, marital status,
   loans).
4. Identify associations behind successful conversions, without claiming causation.
5. Build a leakage-aware model estimating conversion likelihood *before* a call.
6. Turn the analysis into concrete, data-grounded recommendations.

## 5. Honest-Analytics Policy

This dataset has no website visitors, leads, revenue, or ROI fields, so the app never
invents them. Every number shown is one of:

| Label | Meaning |
|---|---|
| **Observed Data** | Read directly from the dataset (e.g., `y == "yes"` counts) |
| **Derived Metrics** | Calculated from observed data (rates, age/duration buckets) |
| **Analytical Associations** | Patterns in this dataset — not proof of causation |
| **Recommendations** | Suggested actions derived from the associations above |

Insight and recommendation text is generated from the *currently filtered* data at
runtime (`generate_insights()`, `generate_recommendations()`) — nothing is hardcoded.

## 6. Funnel Methodology

The dataset has no visitor/lead stages, so the funnel is built only from what exists:

```
Customers Contacted  (observed: campaign ≥ 1, i.e. every record)
        ↓
Engaged Conversations  (derived: call duration ≥ adjustable threshold, default 60s)
        ↓
Subscribed  (observed: y == "yes", among engaged calls)
```

The engagement threshold is a sidebar slider, and the **Funnel Analysis** page includes
a sensitivity chart showing how the middle stage changes as the threshold changes, plus
a reconciliation line so the derived funnel never hides conversions that happened in
shorter calls.

## 7. Data Cleaning

`clean_data()` performs, in order:

1. Column-name normalization (lower-case, stripped, underscored).
2. Schema validation — a CSV missing required columns is rejected with a clear message
   instead of crashing.
3. Text normalization (trim/lower-case) and `"unknown"` fill for blank categoricals.
4. Numeric coercion with sensible fills (`pdays = -1` → "not previously contacted").
5. Target standardization to `yes`/`no`; rows with no usable target are dropped and
   counted.
6. Exact-duplicate removal (counted and reported).
7. Derived analysis buckets: age group, campaign-frequency group, previous-contacts
   group, duration group, pdays group.

All counts from this process are shown in **Data Quality & Export**.

## 8. Dashboard Features

- **Sidebar filters** — job, marital status, education, housing/personal loan, contact
  type, month, previous outcome, age range, balance range, campaign contacts, plus a
  one-click **Reset Filters**. Every KPI, chart, and insight recalculates from the
  filtered data.
- **Executive Dashboard** — KPI cards, funnel, conversion donut, top insights, top
  recommendations.
- **Funnel Analysis** — funnel chart, stage table, conversion progression, and a
  threshold-sensitivity view.
- **Channel & Campaign** — conversion by contact type, previous outcome, month,
  campaign frequency, and a month × channel heatmap.
- **Customer Segmentation** — conversion by job, age group, education, marital status,
  loans/default, plus age and balance distributions by outcome.
- **Campaign Performance** — conversion vs. contact count and call duration, an
  effort-vs-return (diminishing-returns) chart, and distribution views.
- **Customer Insights** — the auto-generated insight feed, grouped by theme.
- **Recommendations** — auto-generated, data-grounded action items.
- **Predictive Insights** — see below.
- **Data Quality & Export** — row/column counts, missing values, duplicates, dtypes,
  target balance, numeric summary, raw preview, plus **Download filtered CSV** and
  **Download summary report** buttons available from the sidebar on every page.

## 9. Predictive Analytics — "Campaign Outcome Prediction"

A Logistic Regression or Random Forest model estimates subscription likelihood using
**only information available before a call is made** (customer profile, planned
contact channel/month, and prior-campaign history). `duration` (call length) is
**excluded by default** and only enabled via an explicit, clearly-labeled checkbox,
because it is only known after a call ends and would leak the outcome into the
features — a model trained on it cannot actually be used to decide who to call next.

Reported for every run: train/test split size, accuracy, precision, recall, F1,
ROC AUC (with a majority-class baseline for context), a confusion matrix, aggregated
feature importance, and a decile-lift chart. All of it is framed as statistical
association, not a causal explanation of why customers subscribe.

## 10. Technology Stack

Python 3 · Streamlit · pandas · NumPy · Plotly · scikit-learn.

## 11. Project Architecture

```text
marketing-funnel-analysis/
├── app.py                 # Streamlit application (single entry point)
├── requirements.txt
├── README.md
├── bank-full.csv          # full dataset (auto-detected)
├── bank.csv               # 10% sample (auto-detected fallback)
├── data/
│   ├── bank-full.csv
│   ├── bank.csv
│   └── bank-names.txt     # original UCI data dictionary / citation
├── .streamlit/
│   └── config.toml        # dashboard theme
├── assets/                # optional logo.png
└── screenshots/           # add dashboard screenshots here for your submission
```

`app.py` is organized into small, reusable functions: `load_data`, `clean_data`,
`calculate_kpis`, `calculate_funnel`, `apply_filters`, `rate_table`,
`generate_insights`, `generate_recommendations`, `train_model`, and a set of chart
builders — no one giant script, no hardcoded statistics.

## 12. Installation & Running Locally

```bash
git clone YOUR_REPOSITORY_URL
cd marketing-funnel-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open it in your
browser. If no CSV is found next to `app.py`, use the sidebar **Upload a Bank Marketing
CSV** control; both comma- and semicolon-delimited files are supported, and
`bank-full.csv`/`bank.csv` schema differences are handled automatically.

## 13. Deploying to Streamlit Community Cloud

1. Push this folder to a public (or Streamlit-linked private) GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repository, branch, and set **Main file path** to
   `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.
5. Once live, copy the app URL for your Task 3 submission.

## 14. Uploading to GitHub

```bash
cd marketing-funnel-analysis
git init
git add .
git commit -m "Marketing Funnel & Conversion Performance Analysis - Task 3"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

## 15. Key Insights (example — regenerated live from the data in-app)

- Observed conversion rate across all 45,211 contacted customers is **11.70%**.
- The `cellular` channel shows a higher observed conversion rate than `telephone` or
  unrecorded (`unknown`) contact types.
- Customers contacted 6+ times convert at a markedly lower rate than those contacted
  1–2 times — consistent with diminishing returns from repeated contact, not evidence
  that calling more *causes* refusal.
- Customers with a **successful previous campaign outcome** convert far above the
  overall average, making them a strong re-engagement audience.
- Certain jobs (e.g., student, retired) and the 66+ age group show higher observed
  conversion than the overall base.

(Exact figures shift with whatever filters are applied — see the live dashboard.)

## 16. Recommendations (example)

1. Prioritize higher-performing contact channels where operationally feasible.
2. Test a cap on repeated contact attempts rather than assuming more calls help.
3. Re-engage customers with a favorable previous-campaign outcome first.
4. Segment campaigns and messaging by job, age, and education.
5. Pilot additional outreach in historically higher-converting months.
6. Improve capture of contact-channel data where it is currently unrecorded.
7. Validate every association above with a controlled experiment before scaling it.

## 17. Limitations & Future Improvements

- The dataset has no visitor/lead/revenue data, so funnel stages above "contacted" are
  intentionally limited to what can be derived from call records.
- `duration` is a strong predictor but is a post-call variable; a production model
  should rely on it only for post-call analytics, not for deciding who to call.
- Future work: time-series view of campaign waves across the 2008–2010 period,
  cost/ROI modeling if cost data becomes available, and A/B test tracking for the
  recommendations above.

## 18. Screenshots

Add screenshots of each dashboard page to `screenshots/` for your submission
(e.g., `executive.png`, `funnel.png`, `segments.png`, `predictive.png`).

## 19. Citation

> S. Moro, R. Laureano and P. Cortez. *Using Data Mining for Bank Direct Marketing: An
> Application of the CRISP-DM Methodology.* In P. Novais et al. (Eds.), Proceedings of
> the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães,
> Portugal, October 2011. EUROSIS.

## 20. Author

Prepared for **Future Interns — Data Science & Analytics, Task 3 (2026)**.
