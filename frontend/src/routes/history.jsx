import { useEffect, useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { AlertTriangle, ArrowLeft, Clock, MapPin, RefreshCw } from "lucide-react";
import { getHistory } from "@/api";
import { Button } from "@/components/ui/button";
import SeasonalPrediction from "@/components/seasonal-prediction";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Anomaly History | AWS Network Monitor" },
      { name: "description", content: "Live anomaly history from the SkyGuard backend sensor database." },
      { property: "og:title", content: "Anomaly History | AWS Network Monitor" },
      { property: "og:description", content: "Live anomaly records across weather stations with location and time frame." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: HistoryPage,
});

function mapHistoryRow(row) {
  const status = String(row.status || "NORMAL").toUpperCase();
  const fault = String(row.fault_type || "");
  const category = String(row.anomaly_category || "");
  const parameter = fault && fault !== "None"
    ? fault.replace(/ Sensor$/i, "")
    : category.includes("PRESSURE")
      ? "Pressure"
      : category.includes("HUMIDITY")
        ? "Humidity"
        : category.includes("TEMP")
          ? "Temperature"
          : "Weather / sensor";

  let value = null;
  if (parameter.toLowerCase().includes("pressure")) value = row.pressure;
  else if (parameter.toLowerCase().includes("humidity")) value = row.humidity;
  else value = row.temperature;

  return {
    sensorId: String(row.sensor_id),
    location: row.locality || row.city || row.district || String(row.sensor_id),
    state: row.state || "Unknown",
    lat: Number(row.latitude),
    lng: Number(row.longitude),
    parameter,
    reading: value == null ? "—" : Number(value).toFixed(2),
    // Preserve raw sensor fields for the seasonal dashboard.
    temperature: row.temperature ?? null,
    pressure: row.pressure ?? null,
    humidity: row.humidity ?? null,
    field_predictions: row.field_predictions ?? null,
    status,
    severity: row.severity || "NONE",
    confidence: row.confidence == null ? null : Number(row.confidence),
    health: row.health == null ? null : Number(row.health),
    start: row.observation_time || row.created_at || "",
    explanation: row.explanation || category.replaceAll("_", " ") || "Backend observation",
    // Preserve backend seasonal-prediction fields for the History dashboard.
    suggested_value: row.suggested_value ?? null,
    prediction_field: row.prediction_field ?? null,
    prediction_confidence: row.prediction_confidence ?? null,
    prediction_confidence_label: row.prediction_confidence_label ?? null,
    prediction_components: row.prediction_components ?? null,
    prediction_weights: row.prediction_weights ?? null,
    prediction_neighbor_count: row.prediction_neighbor_count ?? null,
    seasonal_points_used: row.seasonal_points_used ?? null,
    seasonal_uses_day_of_year: row.seasonal_uses_day_of_year ?? false,
    seasonal_uses_day_of_week: row.seasonal_uses_day_of_week ?? false,
    seasonal_uses_month: row.seasonal_uses_month ?? false,
    seasonal_history_span_days: row.seasonal_history_span_days ?? null,
    seasonal_strength: row.seasonal_strength ?? null,
    seasonal_strength_label: row.seasonal_strength_label ?? null,
    trend_points_used: row.trend_points_used ?? null,
    trend_sequence: row.trend_sequence ?? null,
    prediction_nearest_distance_km: row.prediction_nearest_distance_km ?? null,
  };
}

function HistoryPage() {
  const [filter, setFilter] = useState("all");
  const [stateFilter, setStateFilter] = useState("all");
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      setError("");
      const data = await getHistory();
      const rows = Array.isArray(data?.history) ? data.history.map(mapHistoryRow) : [];
      setRecords(rows);
    } catch (err) {
      console.error(err);
      setError("Backend unavailable — start FastAPI on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 30000);
    return () => window.clearInterval(timer);
  }, []);

  const sensors = useMemo(() => Array.from(new Set(records.map((r) => r.sensorId))).sort(), [records]);
  const states = useMemo(() => Array.from(new Set(records.map((r) => r.state))).sort(), [records]);
  const rows = records.filter((record) =>
    (filter === "all" || record.sensorId === filter) &&
    (stateFilter === "all" || record.state === stateFilter),
  );
  const active = rows.filter((r) => r.status === "SENSOR_ANOMALY").slice(0, 10);

  // Seasonal prediction is shown for the currently filtered sensor.
  // When no sensor is selected, use the most recent backend records.
  const seasonalRows = useMemo(() => {
    const source = filter === "all" ? records : records.filter((record) => record.sensorId === filter);
    return source
      .slice()
      .sort((a, b) => String(b.start).localeCompare(String(a.start)));
  }, [records, filter]);

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="flex min-h-14 flex-wrap items-center justify-between gap-3 border-b border-border bg-surface px-4 py-2">
        <div className="flex items-center gap-3">
          <Link to="/" className="inline-flex size-9 items-center justify-center rounded-md hover:bg-accent" aria-label="Back to map">
            <ArrowLeft className="size-4" />
          </Link>
          <div>
            <h1 className="text-sm font-bold uppercase tracking-wide">Anomaly History</h1>
            <p className="text-[10px] text-muted-foreground">Live records from the backend database</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)} className="h-9 rounded-sm border border-border bg-surface px-2 text-xs" aria-label="Filter by state">
            <option value="all">All states</option>
            {states.map((state) => <option key={state} value={state}>{state}</option>)}
          </select>
          <select value={filter} onChange={(e) => setFilter(e.target.value)} className="h-9 rounded-sm border border-border bg-surface px-2 text-xs" aria-label="Filter by sensor">
            <option value="all">All sensors</option>
            {sensors.map((sensor) => <option key={sensor} value={sensor}>{sensor}</option>)}
          </select>
          <Button variant="outline" size="sm" onClick={load} disabled={loading} title="Refresh history"><RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} /></Button>
        </div>
      </header>

      {error && <div className="mx-auto max-w-6xl px-4 pt-4 text-xs text-destructive">{error}</div>}

      <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
        <section>
          <div className="section-label critical-label"><span>Critical sensor anomalies</span><b>{active.length}</b></div>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {active.length ? active.map((record) => (
              <div key={`${record.sensorId}-${record.start}-${record.parameter}`} className="rounded-sm border border-destructive/40 bg-surface p-4 shadow-panel">
                <div className="flex items-center gap-2 text-destructive"><AlertTriangle className="size-4" /><b className="font-mono text-sm">{record.sensorId}</b></div>
                <p className="mt-2 text-sm font-semibold">{record.parameter} reading {record.reading}</p>
                <p className="mt-1 flex items-center gap-1 text-xs text-muted-foreground"><MapPin className="size-3" />{record.location} · {record.state} · {record.lat.toFixed(3)}, {record.lng.toFixed(3)}</p>
                <p className="mt-1 flex items-center gap-1 text-xs text-muted-foreground"><Clock className="size-3" />{record.start}</p>
                <p className="mt-2 text-xs text-muted-foreground">{record.explanation}</p>
              </div>
            )) : <p className="text-xs text-muted-foreground">No critical anomaly records for the selected filters.</p>}
          </div>
        </section>

        <section>
          <div className="section-label"><span>Backend history</span><b>{loading ? "…" : rows.length}</b></div>
          <div className="mt-3 overflow-x-auto rounded-sm border border-border bg-surface">
            <table className="w-full text-left text-xs">
              <thead className="bg-subtle text-[11px] uppercase tracking-wide text-muted-foreground">
                <tr>{["Sensor", "State / Location", "Coordinates", "Parameter", "Reading", "Status", "Health", "Time"].map((heading) => <th key={heading} className="px-3 py-2 font-semibold">{heading}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-border">
                {rows.map((record) => (
                  <tr key={`${record.sensorId}-${record.start}-${record.parameter}`}>
                    <td className="px-3 py-2 font-mono font-bold">{record.sensorId}</td>
                    <td className="px-3 py-2">{record.state} / {record.location}</td>
                    <td className="px-3 py-2 font-mono text-muted-foreground">{record.lat.toFixed(3)}, {record.lng.toFixed(3)}</td>
                    <td className="px-3 py-2">{record.parameter}</td>
                    <td className="px-3 py-2 font-mono font-bold">{record.reading}</td>
                    <td className="px-3 py-2 font-mono">{record.status}</td>
                    <td className="px-3 py-2 font-mono">{record.health == null ? "—" : `${record.health.toFixed(0)}%`}</td>
                    <td className="px-3 py-2 font-mono">{record.start}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <SeasonalPrediction rows={seasonalRows} />
      </div>
    </main>
  );
}
