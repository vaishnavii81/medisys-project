# MediSys — Enhanced Healthcare Management System

## Quick Start

```bash
cd medisys
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000 in your browser.
Sign up for a new account, then log in.

## What's New (vs original)

| Feature | Original | MediSys |
|---|---|---|
| Authentication | Plain-text passwords | SHA-256 hashed passwords |
| Patient Records | None | Full CRUD (add/view/edit/delete) |
| Dashboard | None | Stats: total, high risk, low risk, appointments |
| Search & Filter | None | Filter patients by name, risk level |
| Appointments | None | Schedule, view, cancel appointments |
| MediBot Chatbot | None | Rule-based health assistant (12+ topics) |
| Reports | None | Summary stats + full patient table |
| CSV Export | None | One-click export of all patient records |
| UI | Basic inline HTML/CSS | Modern sidebar layout, DM Sans font, card system |
| Prediction UI | Simple form | Enhanced with field guide + result alerts |

## Files
- `app.py` — Main Flask application (all features)
- `heart_disease.csv` — Original UCI dataset (unchanged)
- `medisys.db` — SQLite database (auto-created on first run)
- `requirements.txt` — Python dependencies
