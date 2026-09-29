# SkyGuard AI — Complete Package

SkyGuard AI is a FastAPI + React/Vite Automatic Weather Station monitoring system with anomaly detection, spatial comparison, prediction support, maintenance analysis, history, and a data-grounded assistant.

## Backend

From `backend`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit backend/.env and put your real Windy API key in WINDY_API_KEY
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Put your Windy API key in `backend/.env` (or in an environment variable named `WINDY_API_KEY`):

```text
WINDY_API_KEY=your_key_here
```

The backend loads and normalizes the registered sensors dynamically. There is no five-sensor limit.

## Frontend

From `frontend`:

```bash
npm install
npm run dev
```

Optional frontend environment file:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The frontend source is JavaScript/JSX and uses Vite.

## Run the full system

1. Start the backend on `127.0.0.1:8000`.
2. Start the frontend with `npm run dev`.
3. Open the URL shown by Vite.

## Important data behavior

- `temperature`, `pressure`, and `humidity` remain the original received values.
- Prediction/suggestion values are stored separately and never overwrite the raw readings.
- Windy's model-valid timestamp is stored separately as `weather_model_time`.
- `timestamp`, `observation_time`, and `actual_fetch_time` represent the actual polling time used by the anomaly timing logic.
- API failures are preserved as `DATA_UNAVAILABLE` records with `api_status` and `api_error`.
- Sensor IDs are canonicalized to uppercase underscore format, e.g. `AWS-006` becomes `AWS_006`.

## Core logic policy

The anomaly engine's detection rules and decision hierarchy are preserved. The stabilization work in this package is focused on integration errors, timestamp handling, database persistence, sensor-ID consistency, API failure handling, and missing frontend/backend data fields.
