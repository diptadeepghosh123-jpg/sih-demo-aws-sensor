# SkyGuard 8-hour stabilization pass — final fixes

## Fixed in this pass

1. **Temperature sudden-change escalation**
   - Large isolated temperature steps (>15% within 10 minutes) now escalate to `SENSOR_ANOMALY`.
   - Regional support is still checked first so widespread movement can remain `WEATHER_EVENT`.

2. **Humidity sudden-change escalation**
   - A sudden latest humidity step (>=10 RH points within 10 minutes) is treated as `SENSOR_ANOMALY` when regional agreement does not support a weather event.
   - Existing 5-minute, persistence, orderly-pattern, and regional-weather logic remains intact.

3. **Seasonal prediction database persistence**
   - Added persistent fields for:
     - prediction confidence label
     - seasonal strength label
     - trend sequence
     - nearest prediction-neighbor distance
   - Existing SQLite databases are migrated automatically; they are not deleted.

4. **Supporting ML fault classifier**
   - Added `predict_fault_type()` with a separate supporting Random Forest model.
   - Outputs `NORMAL`, `WEATHER_EVENT`, `SUDDEN_SPIKE`, `CALIBRATION_DRIFT`, or `ISOLATED_SENSOR_FAULT` as supporting evidence only.
   - `anomaly.py` remains authoritative for final status/fault decisions.

5. **Prediction dashboard contract**
   - History mapping now preserves the new prediction metadata.
   - Seasonal Prediction details can display labels, nearest distance, and stored trend sequence.

## Verified

- Backend Python compilation: PASS
- FastAPI startup/health: PASS
- Sensor registry endpoint: PASS
- `/api/analyze`: PASS
- API failure preservation: PASS
- Temperature sudden-step test: `SENSOR_ANOMALY`
- Humidity sudden-step test: `SENSOR_ANOMALY`
- Fault-type ML classifier: PASS
- Database insert with 35 prediction/anomaly parameters: PASS
- Prediction metadata round-trip: PASS

## Environment note

The frontend source and lockfile are included. A full `npm ci` / browser build could not be completed inside the isolated build environment because dependency installation timed out. Run `npm install` and `npm run build` or `npm run dev` on the demo machine.


## Chatbot integration
- Added /api/chat operational assistant endpoint.
- Added SkyGuard Assistant floating frontend panel.
- Assistant answers from sensor registry, stored history, prediction metadata, and maintenance data without requiring an external AI key.

## 2026-09-28 integration/bug-fix pass

- Updated `weather_api.py` to keep Windy's model-valid timestamp separate from the actual polling timestamp used by anomaly timing.
- Preserved API failure rows with explicit API status/error metadata.
- Canonicalized sensor IDs across registry, anomaly input, API boundaries, and database persistence (`AWS-006` -> `AWS_006`).
- Added safe SQLite migrations for live API metadata and anomaly explanation/evidence fields.
- Preserved `weather_model_time` and `actual_fetch_time` in analysis/database records without changing raw weather values.
- Fixed history ordering ties by using the database row ID as a secondary newest-first key.
- Repaired frontend sensor-ID canonicalization so add/remove/history operations stay consistent.
- Added root setup documentation and backend `.env.example` without embedding secrets.
- Removed incomplete environment-specific `node_modules` from the distributable package.

### Validation completed

- Python backend compilation: PASS
- Windy fetch simulation with 8 registered sensors: PASS
- Sensor ID canonicalization and duplicate protection: PASS
- Timestamp separation / 60-second anomaly gap test: PASS
- Analyze -> SQLite save -> history round-trip: PASS
- FastAPI `/`, `/api/health`, `/api/sensors`, `/api/analyze`, `/api/history`, `/api/chat`: PASS
- ZIP integrity test: PASS

The core anomaly-detection rules and decision hierarchy were not rewritten.
