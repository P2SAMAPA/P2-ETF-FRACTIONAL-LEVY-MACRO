# Fractional Lévy Process (FLP) with Macro Intensity

Models ETF returns as a fractional Lévy process where jump intensity depends on macro variables (VIX, DXY, yields). The per‑ETF score is the expected jump return = λ(macro) × μⱼ – a macro‑informed signal for large moves.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Jump detection via threshold (|return| > 2%)
- Logistic regression to model jump probability from macro variables
- Score = predicted jump probability × expected jump size
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-fractional-levy-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n) per ETF)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is likely to experience a large positive jump given current macro conditions.
- Low or negative score → unlikely to jump or expected jump is negative.

## Requirements

See `requirements.txt`.
