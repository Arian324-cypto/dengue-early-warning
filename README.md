# Dengue Early-Warning System (Team DEBUG, EDU AI Engineering Hackathon 2026, Phase 1)

Predicts ward-level dengue risk (Low / Medium / High / Critical) **2 to 4 weeks ahead** from weather (temperature, humidity, rainfall with multi-week lags)
and ward vulnerability, with a Streamlit dashboard showing the forecast, an 80% range, the main drivers and a ward heatmap.
Design, validation split and cost controls: see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start (under 3 minutes)
Requires Python 3.10+.

```bash
git clone https://github.com/Arian324-cypto/dengue-early-warning.git
cd dengue-early-warning
pip install -r requirements.txt
streamlit run app.py
```

The trained model (`models/dengue_model.joblib`) and data (`data/dengue_weekly.csv`) are included, so the app starts immediately.
On Windows use `python -m pip install -r requirements.txt` and `python -m streamlit run app.py` if `pip`/`streamlit` is not recognised.

To retrain from scratch (about 3 seconds): `python src/train.py`

## What is in the repo
| Path | Purpose |
|---|---|
| `src/pipeline.py` | per-ward rainfall/temperature/humidity lag features |
| `src/train.py` | chronological split, training, metrics, joblib export |
| `src/predict.py` | fast inference, risk tiers, driver breakdown |
| `app.py` | Streamlit dashboard (upload your own weekly CSV in the sidebar) |
| `models/` | serialized model, `metrics.json`, test predictions |
| `data/dengue_weekly.csv` | competition dataset |

## Results (held-out 2024-2025)
MAE 0.83 cases/week (seasonal baseline 0.91, ward-mean baseline 2.95). Tier accuracy 83%. See ARCHITECTURE.md for limitations.

## API usage
No external APIs are called; no API key is needed.

## AI assistant disclosure
* **Claude (Anthropic, Claude Sonnet 5.5)**: used by the team for guidance and to generate/explain the code, README and ARCHITECTURE.md. The team reviewed, ran and understands the code.
* No other AI assistants and no external APIs were used.

## Phase 2 (onsite final): operations command system

The same app now turns the forecast into action. Tabs: **Forecast**, **Resource allocation** (saline, kits, beds, crews under a
30% to 100% stock slider, deficit alerts), **Hospital overflow** (same-city patient routing, net deficits, utilisation),
**What-if simulator** (deploy crews, see weeks 2 to 4, ROI per crew), **Dispatch notices** (SMS, WhatsApp, printable brief, English and Bangla).

```
streamlit run app.py                 # dashboard
python src/export.py                 # writes sample SMS / WhatsApp / HTML briefs to exports/
python src/offline_check.py          # offline kill-switch test (blocks the network, runs everything)
python tests/test_ops.py             # checks on allocation, routing and the simulator
```

Everything runs locally with no API keys; details and stated assumptions are in `ARCHITECTURE.md`.
Commit messages for this phase start with `version-onsite-final:`.

## Team
Arian (team leader), Tajun, Sami. Each member's work is in the Git history.
