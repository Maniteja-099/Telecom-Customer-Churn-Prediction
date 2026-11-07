"""Generate deterministic synthetic telecom data for the demo project."""
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "data" / "telecom_customers.csv"


def main():
    rng = np.random.default_rng(42)
    n = 2500
    tenure = rng.integers(0, 73, size=n)
    monthly = np.clip(rng.normal(72, 23, size=n), 18, 125).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.48, 0.38, 0.14])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n)
    support = rng.choice(["Yes", "No"], n, p=[0.32, 0.68])
    security = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    paperless = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])

    # Illustrative churn signal; deliberately not intended to mimic a real provider.
    log_odds = (
        -1.45
        + 1.05 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        - 0.025 * tenure
        + 0.012 * (monthly - 70)
        + 0.55 * (payment == "Electronic check")
        - 0.50 * (support == "Yes")
        - 0.42 * (security == "Yes")
        + rng.normal(0, 0.4, n)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    churn = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "tech_support": support,
            "online_security": security,
            "paperless_billing": paperless,
            "churn": churn,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Wrote {len(frame):,} synthetic customers to {OUT}")


if __name__ == "__main__":
    main()
