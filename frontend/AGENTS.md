# SkyGuard AI Frontend Notes

- Keep the frontend in JavaScript/JSX.
- Preserve the existing AWS Network Monitor visual structure and interaction model when making changes.
- Live sensor values, statuses, anomaly categories, locations, and history must come from the FastAPI backend whenever the corresponding endpoint exists.
- Do not replace backend anomaly decisions with frontend-only simulation logic.
- Client-side sensor placement/deletion is a dashboard-view operation unless a backend registry endpoint is available.
- Keep API integration isolated in `src/api.js`.
