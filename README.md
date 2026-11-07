# Telecom Customer Churn Prediction

A full-stack machine-learning project that estimates which telecom customers may leave, shows churn patterns in a React dashboard, and exposes predictions through a FastAPI backend.

## Features

- Reproducible sample telecom customer data (generated locally; no personal data or external download).
- A scikit-learn preprocessing and classification pipeline that handles numeric and categorical fields.
- Model evaluation with accuracy, precision, recall, F1, ROC AUC, and a confusion matrix.
- React dashboard for summary metrics, churn patterns, and single-customer scoring.
- FastAPI endpoints for customer metrics and churn probability estimates.

## Setup

Requires Python 3.10 or newer and Node.js 18 or newer.

```bash
cd telecom-customer-churn
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:8501`. The backend API is at `http://localhost:8000` and its interactive documentation is at `http://localhost:8000/docs`.

The first run generates `data/telecom_customers.csv` and saves the trained pipeline and metrics under `artifacts/`. These generated files are excluded from version control. The data is synthetic and intended for learning, not business decisions.

## Project files

- `generate_data.py`: creates a deterministic, labeled sample dataset.
- `train.py`: trains and evaluates a reproducible classification pipeline.
- `backend/main.py`: FastAPI metrics and prediction endpoints.
- `frontend/`: React application built with Vite.
- `src/`: shared data, model, and scoring functions.

## Dataset fields

The sample contains customer tenure, monthly charges, contract type, internet service, payment method, support and security add-ons, and a churn label. `customer_id` is an identifier and is excluded from model training.

## Limitations

The generated data encodes illustrative relationships and does not represent a real telecom population. Risk scores are model estimates, not certainty or causal explanations. Validate on representative, consented data before any operational use.
