# Maritime Passenger Survival API

A machine learning REST API that predicts whether a maritime passenger would survive, based on passenger attributes.

Live API: [https://maritime-survival-api.onrender.com](https://maritime-survival-api.onrender.com)

---

## Endpoints

### `GET /`
Health check.

```json
{ "message": "Maritime Survival Prediction API is running!" }
```

### `POST /predict`
Predict survival for a passenger. Rate limited to **10 requests/minute**.

Request body:
```json
{
  "Age": 25,
  "Gender": "male",
  "BoardingPort": "S",
  "Title": "Mr",
  "TicketTier": 3,
  "TicketCost": 10.0,
  "FamilySize": 1,
  "FarePerPerson": 10.0,
  "RelativesAboard": 0,
  "ParentsChildren": 0,
  "CLass": "A123"
}
```

Response:
```json
{
  "survived": false,
  "survival_probability": 0.18,
  "message": "Did not survive"
}
```

---

## Tech Stack

- FastAPI + Uvicorn
- scikit-learn + XGBoost
- SlowAPI (rate limiting)
- Deployed on Render

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run Tests

```bash
pytest
```

---

## CI/CD

GitHub Actions workflow (`.github/workflows/deploy.yml`) runs on every push to `main`:
1. Installs dependencies
2. Runs `pytest`
3. Triggers a Render deploy via webhook
