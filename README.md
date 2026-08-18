# NOPE-IN: Neural Options Pricing Engine — India

**A regime-conditioned residual learning framework for NSE derivative markets.**

> *Is the price the market quotes for an option rational? And if not, what exactly does it get wrong — and can a neural network learn that?*

| Field | Value |
|---|---|
| **Project** | NOPE-IN (Neural Options Pricing Engine — India) |
| **Version** | 1.0 |
| **Market** | NSE — Index Options (NIFTY 50, BANKNIFTY) |
| **Model** | Regime-Conditioned Residual Neural Network (MoE) |
| **Novel Contribution** | BSM error anatomy in Indian markets + HMM regime gating + conformal prediction |
| **Status** | Phase 0 + Phase 1 complete — Phase 2 feature engineering implemented |

---

## Problem Statement

Black-Scholes-Merton (BSM) is the canonical option pricing model, but its assumptions are systematically violated in real markets — especially in India:

- **Weekly expiry concentration** — NSE weekly NIFTY options create aggressive, non-linear theta decay
- **Retail-dominated order flow** — liquidity-driven IV spikes that BSM cannot model
- **India VIX** — a first-class regime signal with no published use in neural options pricing
- **Scheduled macro events** — RBI MPC, Union Budget, FOMC spillover create predictable volatility clusters

**Core approach — residual learning:**

```
Final Price = BSM_Price + Neural_Net(Features → BSM_Error_Estimate)
```

We do not replace BSM; we learn where, when, and by how much it misprices NIFTY and BANKNIFTY options.

---

## Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│   Stage 1    │    │   Stage 2    │    │       Stage 3        │
│  DATA        │───▶│  FEATURE     │───▶│   REGIME DETECTION   │
│  INGESTION   │    │  ENGINEERING │    │   (HMM · 4 states)   │
└──────────────┘    └──────────────┘    └──────────────────────┘
                                                   │
┌──────────────┐    ┌──────────────┐    ┌────────▼─────────────┐
│   Stage 5    │    │   Stage 4b   │    │       Stage 4a       │
│  EXPLAIN-    │◀───│  CONFORMAL   │◀───│   NEURAL PRICING     │
│  ABILITY     │    │  CALIBRATION │    │   (MoE · 4 experts)  │
└──────────────┘    └──────────────┘    └──────────────────────┘
```

---

## Instruments in Scope

| Instrument | Expiry | Priority |
|---|---|---|
| NIFTY 50 Index Options | Weekly (Thu) + Monthly | **Primary** |
| BANKNIFTY Index Options | Weekly (Wed) + Monthly | **Primary** |
| FINNIFTY Index Options | Tuesday weekly | Secondary |

**Practical data start date:** 2018 (weekly options liquidity). Full history available from NSE Bhav Copy back to 2004.

---

## Data Sources

| Dataset | Source | Purpose |
|---|---|---|
| Options chain | NSE F&O Bhav Copy | Primary training target (`SETTLE_PR`) |
| Underlying OHLCV | yfinance | NIFTY / BANKNIFTY spot prices |
| India VIX | NSE website | Regime conditioning |
| Risk-free rate | RBI DBIE | BSM discounting |
| Events calendar | Manual curation | RBI MPC, Budget, FOMC, expiry proximity |
| IV surface | DIY from Bhav Copy | Surface stats + CNN embedding |

---

## Planned Repository Structure

```
nope-in/
├── configs/          # Hydra YAML (data, model, regime, features, training)
├── data/
│   ├── raw/          # Bhav Copy, OHLCV, VIX, rates (never modify)
│   ├── manual/       # Events calendar, curated RBI data
│   └── processed/    # Cleaned parquet outputs
├── src/
│   ├── data/         # Fetchers, validators, pipeline
│   ├── features/     # BSM, vol surface, feature store
│   ├── regime/       # HMM fitting + labelling
│   ├── models/       # NOPE-IN MoE, conformal, baselines
│   ├── training/     # PyTorch Lightning trainer
│   └── evaluation/   # Metrics, SHAP, model comparison
├── notebooks/        # EDA → results (00–06)
├── dashboard/        # Streamlit app
├── scripts/          # Pipeline entry points
└── tests/            # BSM property tests, schema, conformal coverage
```

---

## Implementation Phases

| Phase | Goal | Duration |
|---|---|---|
| **0 — Setup** | Poetry, DVC, Hydra configs, Makefile, pre-commit | 2–3 days |
| **1 — Data** | Bhav Copy fetcher, yfinance, RBI rates, events calendar | 5–7 days |
| **2 — Features** | BSM pricer, IV surface, feature store (~36–108 features) | 5–7 days |
| **3 — Regime** | HMM (4 states) on India VIX + realised vol | 3–4 days |
| **4 — Model** | NOPE-IN MoE training + conformal calibration | 5–7 days |
| **5 — Eval** | Metrics, SHAP, dashboard, article-ready plots | 4–5 days |

---

## Quick Start (Phase 0 + Phase 1)

```bash
cd nope-in
python3 -m venv .venv
source .venv/bin/activate
pip install -e . pytest

# Tier 0 smoke test (Jan 2023, ~22 trading days)
make data-smoke

# Tier 1 full download (2018–2023, resumable)
make data

# Validate processed outputs
make validate

# Phase 2 — feature engineering (smoke: 1 month; full: all data, ~30–60 min)
make features-smoke
make features

# Run tests
make test
```

Processed outputs land in `data/processed/`:
- `options_chain.parquet` — NIFTY/BANKNIFTY index options (OPTIDX)
- `underlying_ohlcv.parquet` — NIFTY + BANKNIFTY daily OHLCV
- `india_vix.parquet` — India VIX with basic features
- `rates.parquet` — RBI rates (from `data/manual/rbi_rates_seed.csv`)
- `events_calendar.parquet` — NSE expiries + RBI/Budget/FOMC proximity features

Raw bhav copy CSVs: `data/raw/bhav_copy/{YYYY}/{MM}/`


## Evaluation Targets

| Model | Expected ATM RMSE |
|---|---|
| BSM (baseline) | ₹4.5–₹8.0 |
| Heston | ₹3.2–₹6.0 |
| XGBoost | ₹2.8–₹5.0 |
| **NOPE-IN (target)** | **₹1.8–₹3.5** (>30% improvement over BSM) |

**Walk-forward split:** Train 2018–2021 · Val 2022 · Test 2023–2024 (OOS, sealed)

---

## Research Contributions

1. Systematic BSM error decomposition for NSE weekly options
2. India VIX as a gating feature for a neural pricing MoE
3. Weekly expiry BSM failure modes (pinning, gamma explosion)
4. RBI/Budget event features integrated into options pricing
5. Conformal prediction for calibrated uncertainty on Indian options
6. DIY IV surface reconstruction from NSE Bhav Copy

---

## License

TBD — research / portfolio project.

---

## References

- PRD: `NOPE_IN_PRD_v1.0.md` (internal)
- NSE F&O Bhav Copy: [nseindia.com/reports-data/fo-derivatives](https://www.nseindia.com/reports-data/fo-derivatives)
