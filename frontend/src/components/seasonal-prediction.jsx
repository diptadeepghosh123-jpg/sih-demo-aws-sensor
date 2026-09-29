import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChevronDown, ChevronUp, Leaf, Target } from "lucide-react";

const PARAMETERS = [
  { key: "temperature", label: "Temperature", unit: "°C", digits: 1 },
  { key: "pressure", label: "Pressure", unit: "hPa", digits: 1 },
  { key: "humidity", label: "Humidity", unit: "%", digits: 1 },
];

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseObject(value) {
  if (!value) return null;
  if (typeof value === "object") return value;

  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
      return null;
    }
  }

  return null;
}

function predictionForRow(row, field) {
  const predictionField = String(row?.prediction_field || "")
    .trim()
    .toLowerCase();

  // New all-field prediction object persisted by database.py.
  const fieldPredictions = parseObject(row?.field_predictions);
  const fieldPrediction = fieldPredictions?.[field];

  if (fieldPrediction?.prediction_available === false) {
    return null;
  }

  const fieldExpected = numberOrNull(fieldPrediction?.expected_value);
  if (fieldExpected != null) return fieldExpected;

  const fieldSuggested = numberOrNull(fieldPrediction?.suggested_value);
  if (fieldSuggested != null) return fieldSuggested;

  // Backward-compatible support for older database records.
  const direct = numberOrNull(row?.[`suggested_${field}`]);
  if (direct != null) return direct;

  const generic = numberOrNull(row?.suggested_value);
  if (generic != null && predictionField === field) return generic;

  const nestedPrediction = parseObject(row?.prediction);
  const nested = numberOrNull(nestedPrediction?.[field]);
  return nested != null ? nested : null;
}

function actualForRow(row, field) {
  const direct = numberOrNull(row?.[field]);
  if (direct != null) return direct;

  const predictionField = String(row?.prediction_field || "")
    .trim()
    .toLowerCase();

  if (predictionField !== field) return null;

  const fieldPredictions = parseObject(row?.field_predictions);
  return numberOrNull(fieldPredictions?.[field]?.prediction_actual_value);
}

function latestPrediction(rows, field) {
  for (const row of rows) {
    const expected = predictionForRow(row, field);
    const actual = actualForRow(row, field);
    if (expected != null || actual != null) {
      return { actual, expected, row };
    }
  }
  return { actual: null, expected: null, row: null };
}

function formatValue(value, parameter) {
  if (value == null) return "—";
  return `${value.toFixed(parameter.digits)} ${parameter.unit}`;
}

function metadata(rows) {
  const source = rows.find((row) =>
    row?.seasonal_strength != null ||
    row?.prediction_confidence != null ||
    row?.seasonal_points_used != null ||
    row?.seasonal_history_span_days != null ||
    row?.trend_points_used != null,
  );

  if (!source) {
    return {
      strength: null,
      strengthLabel: null,
      confidence: null,
      confidenceLabel: null,
      points: null,
      span: null,
      trend: null,
      nearestDistance: null,
      trendSequence: [],
    };
  }

  const strength = numberOrNull(source.seasonal_strength);
  const confidenceNumber = numberOrNull(source.prediction_confidence);
  const confidenceText = source.prediction_confidence == null
    ? null
    : String(source.prediction_confidence);

  let trendSequence = source.trend_sequence;
  if (typeof trendSequence === "string") {
    try { trendSequence = JSON.parse(trendSequence); } catch { trendSequence = []; }
  }
  if (!Array.isArray(trendSequence)) trendSequence = [];

  return {
    strength: strength == null ? null : strength <= 1 ? strength * 100 : strength,
    strengthLabel: source.seasonal_strength_label ?? null,
    confidence: confidenceNumber == null
      ? confidenceText
      : confidenceNumber <= 1 ? confidenceNumber * 100 : confidenceNumber,
    confidenceLabel: source.prediction_confidence_label ?? null,
    points: numberOrNull(source.seasonal_points_used),
    span: numberOrNull(source.seasonal_history_span_days),
    trend: numberOrNull(source.trend_points_used),
    nearestDistance: numberOrNull(source.prediction_nearest_distance_km),
    trendSequence,
  };
}

function buildChartData(rows, field) {
  return rows
    .filter((row) => actualForRow(row, field) != null || predictionForRow(row, field) != null)
    .slice()
    .reverse()
    .map((row, index) => ({
      index: index + 1,
      actual: actualForRow(row, field),
      expected: predictionForRow(row, field),
      time: row?.created_at || row?.observation_time || "",
    }));
}

function MiniPredictionChart({ rows, parameter }) {
  const data = useMemo(() => buildChartData(rows, parameter.key), [rows, parameter.key]);

  if (!data.length) {
    return <div className="flex h-36 items-center justify-center text-xs text-muted-foreground">No prediction history available.</div>;
  }

  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis dataKey="index" hide />
          <YAxis tick={{ fontSize: 9 }} width={42} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 6 }}
            formatter={(value, name) => [
              value == null ? "—" : `${Number(value).toFixed(parameter.digits)} ${parameter.unit}`,
              name === "actual" ? "Actual" : "Expected",
            ]}
            labelFormatter={(index) => data[index - 1]?.time || `Reading ${index}`}
          />
          <Line type="monotone" dataKey="actual" name="actual" dot={false} strokeWidth={2} connectNulls />
          <Line type="monotone" dataKey="expected" name="expected" dot={false} strokeWidth={2} strokeDasharray="5 4" connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function SeasonalPrediction({ rows }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const safeRows = Array.isArray(rows) ? rows : [];
  const stats = useMemo(() => metadata(safeRows), [safeRows]);

  const predictions = useMemo(
    () => PARAMETERS.map((parameter) => ({
      parameter,
      ...latestPrediction(safeRows, parameter.key),
    })),
    [safeRows],
  );

  const hasPrediction = predictions.some((item) => item.expected != null);

  return (
    <section className="rounded-sm border border-border bg-surface shadow-panel">
      <div className="border-b border-border px-4 py-4 sm:px-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Leaf className="size-4" />
              <h2 className="text-sm font-bold uppercase tracking-wide">Seasonal Prediction</h2>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">Overall expected weather values from the backend seasonal predictor.</p>
          </div>
          <Target className="size-5 text-muted-foreground" />
        </div>
      </div>

      {!hasPrediction ? (
        <div className="px-4 py-6 text-xs text-muted-foreground sm:px-5">
          Seasonal prediction is not available yet. The backend needs sufficient historical observations before displaying an expected value.
        </div>
      ) : (
        <>
          <div className="grid gap-px border-b border-border bg-border sm:grid-cols-3">
            {predictions.map(({ parameter, actual, expected }) => (
              <div key={parameter.key} className="bg-surface p-4">
                <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{parameter.label}</p>
                <p className="mt-2 text-xl font-bold tracking-tight">{formatValue(actual, parameter)}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Expected: <span className="font-semibold text-foreground">{formatValue(expected, parameter)}</span>
                </p>
              </div>
            ))}
          </div>

          <div className="border-b border-border p-4 sm:p-5">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wide">Actual vs Expected</h3>
                <p className="mt-1 text-[11px] text-muted-foreground">Recent readings compared with backend expected values.</p>
              </div>
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                <span>━ Actual</span>
                <span>╌ Expected</span>
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-3">
              {PARAMETERS.map((parameter) => (
                <div key={parameter.key} className="rounded-sm border border-border p-3">
                  <p className="mb-1 text-[11px] font-semibold">{parameter.label}</p>
                  <MiniPredictionChart rows={safeRows} parameter={parameter} />
                </div>
              ))}
            </div>
          </div>

          <div className="grid gap-3 p-4 sm:grid-cols-5 sm:p-5">
            <Metric label="Seasonal Strength" value={stats.strength == null ? "—" : `${stats.strength.toFixed(0)}%`} />
            <Metric label="Prediction Confidence" value={typeof stats.confidence === "number" ? `${stats.confidence.toFixed(0)}%` : stats.confidence || "—"} />
            <Metric label="Historical Points" value={stats.points == null ? "—" : stats.points.toLocaleString()} />
            <Metric label="History Span" value={stats.span == null ? "—" : `${stats.span.toFixed(1)} days`} />
            <Metric label="Trend Points" value={stats.trend == null ? "—" : stats.trend.toLocaleString()} />
          </div>

          <div className="border-t border-border px-4 py-3 sm:px-5">
            <button
              type="button"
              onClick={() => setDetailsOpen((value) => !value)}
              className="inline-flex items-center gap-2 text-xs font-semibold hover:underline"
            >
              {detailsOpen ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
              {detailsOpen ? "Hide seasonal details" : "View seasonal details"}
            </button>

            {detailsOpen && (
              <div className="mt-3 grid gap-2 rounded-sm border border-border bg-subtle p-3 text-xs text-muted-foreground sm:grid-cols-3">
                <span>Weekly seasonality: {safeRows[0]?.seasonal_uses_day_of_week ? "Used" : "Not available"}</span>
                <span>Monthly seasonality: {safeRows[0]?.seasonal_uses_month ? "Used" : "Not available"}</span>
                <span>Annual seasonality: {safeRows[0]?.seasonal_uses_day_of_year ? "Used" : "Not available"}</span>
                <span>Seasonal level: {stats.strengthLabel || "—"}</span>
                <span>Confidence level: {stats.confidenceLabel || "—"}</span>
                <span>Nearest sensor: {stats.nearestDistance == null ? "—" : `${stats.nearestDistance.toFixed(2)} km`}</span>
                <span className="sm:col-span-3">Trend sequence: {stats.trendSequence.length ? stats.trendSequence.map((value) => Number(value).toFixed(2)).join(" → ") : "—"}</span>
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-sm border border-border p-3">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm font-bold">{value}</p>
    </div>
  );
}
