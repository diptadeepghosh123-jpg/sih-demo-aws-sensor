# SkyGuard AI — AWS Network Monitor

Real-time Automatic Weather Station monitoring dashboard connected to the SkyGuard AI FastAPI backend.

## Frontend

- React + Vite
- TanStack Router / Start
- Leaflet / React Leaflet
- Recharts
- Tailwind CSS
- JavaScript / JSX

## Run

From the `frontend` directory:

```bash
npm install
npm run dev
```

The frontend reads the backend URL from `VITE_API_BASE_URL` and defaults to:

```text
http://127.0.0.1:8000
```

## Live data flow

```text
AWS / weather data
      ↓
FastAPI backend
      ↓
anomaly detection + spatial analysis + database
      ↓
/api/analyze and /api/history
      ↓
SkyGuard AI dashboard
```

The dashboard refreshes live sensor analysis every 30 seconds and uses backend history for the sensor detail chart and anomaly history page.
