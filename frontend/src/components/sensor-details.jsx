import { useEffect, useMemo, useState } from "react";
import { ArrowLeft } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { getHistory } from "@/api";

function parseTime(value) {
  const date = new Date(value || "");
  return Number.isNaN(date.getTime()) ? value || "" : date;
}

function formatTime(value) {
  const date = parseTime(value);
  if (!(date instanceof Date)) return String(value || "");
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function numeric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function buildTemperatureHistory(rows) {
  const points = rows
    .map((row) => ({
      rawTime: row.observation_time || row.created_at || "",
      time: formatTime(row.observation_time || row.created_at),
      temp: numeric(row.temperature),
      pressure: numeric(row.pressure),
      humidity: numeric(row.humidity),
      status: String(row.status || "NORMAL").toUpperCase(),
      severity: row.severity || "NONE",
      explanation: row.explanation || row.anomaly_category || "",
    }))
    .filter((point) => point.temp !== null)
    .sort((a, b) => new Date(a.rawTime) - new Date(b.rawTime));

  const unique = [];
  const seen = new Set();
  for (const point of points) {
    const key = `${point.rawTime}|${point.temp}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(point);
    }
  }
  return unique.slice(-48);
}

function analyse(points, baseline) {
  const shifts = [];
  for (let i = 1; i < points.length; i += 1) {
    const previous = points[i - 1].temp;
    const current = points[i].temp;
    const step = current - previous;
    const deviation = current - baseline;

    if (Math.abs(step) >= 1.5 || (Math.abs(deviation) >= 2 && Math.abs(previous - baseline) < 2)) {
      const magnitude = Math.abs(deviation);
      shifts.push({
        time: points[i].time,
        delta: deviation,
        step,
        impact: Math.min(100, Math.round((magnitude / Math.max(1, Math.abs(baseline))) * 1000)),
        status: points[i].status,
        text:
          `${points[i].time}: temperature moved ${step >= 0 ? "+" : ""}${step.toFixed(1)}°C from the previous observation` +
          `${points[i].status !== "NORMAL" ? `; backend status is ${points[i].status}.` : "."}`,
      });
    }
  }
  return shifts.slice(-8).reverse();
}

const stateLabel = { normal: "Normal", warning: "Warning", anomaly: "Anomaly" };

export default function SensorDetails({ sensor, onBack }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    getHistory()
      .then((data) => {
        if (!active) return;
        const history = Array.isArray(data?.history) ? data.history : [];
        setRows(history.filter((row) => String(row.sensor_id) === String(sensor.id)));
        setError("");
      })
      .catch((err) => {
        if (!active) return;
        console.error(err);
        setError("Unable to load sensor history from the backend.");
      })
      .finally(() => active && setLoading(false));

    return () => {
      active = false;
    };
  }, [sensor.id]);

  const points = useMemo(() => buildTemperatureHistory(rows), [rows]);
  const baseline = useMemo(() => {
    const values = points.map((point) => point.temp).filter(Number.isFinite);
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }, [points]);
  const current = points.at(-1)?.temp ?? numeric(sensor.raw?.temperature);
  const deviation = baseline !== null && current !== null ? current - baseline : null;
  const shifts = baseline !== null ? analyse(points, baseline) : [];
  const raw = sensor.raw || {};

  return (
    <section className="p-4">
      <Button variant="ghost" size="sm" className="-ml-2 mb-2" onClick={onBack}>
        <ArrowLeft /> All alerts
      </Button>

      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="font-mono text-sm font-bold">{sensor.id}</h3>
          <p className="text-xs text-muted-foreground">{sensor.name}</p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            {raw.state || "Unknown state"} · {raw.district || raw.city || "Unknown area"}
          </p>
        </div>
        <span
          className={`rounded-sm border px-2 py-0.5 text-[11px] font-bold uppercase ${
            sensor.state === "anomaly"
              ? "border-destructive text-destructive"
              : sensor.state === "warning"
                ? "border-sensor-warning text-sensor-warning"
                : "border-primary text-primary"
          }`}
        >
          {stateLabel[sensor.state]}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        {[
          ["Temperature", current == null ? "—" : `${current.toFixed(1)}°C`],
          ["Pressure", numeric(raw.pressure) == null ? "—" : `${numeric(raw.pressure).toFixed(1)} hPa`],
          ["Humidity", numeric(raw.humidity) == null ? "—" : `${numeric(raw.humidity).toFixed(1)}%`],
        ].map(([label, value]) => (
          <div key={label} className="rounded-sm border border-border bg-subtle p-2">
            <div className="text-[10px] uppercase text-muted-foreground">{label}</div>
            <div className="font-mono text-xs font-bold">{value}</div>
          </div>
        ))}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-center">
        {[
          ["Health", raw.health == null ? "—" : `${Number(raw.health).toFixed(0)}%`],
          ["Confidence", raw.confidence == null ? "—" : `${Number(raw.confidence).toFixed(0)}%`],
          ["Neighbors", raw.neighbor_count == null ? "—" : String(raw.neighbor_count)],
          ["Category", raw.anomaly_category || "NORMAL"],
        ].map(([label, value]) => (
          <div key={label} className="rounded-sm border border-border bg-subtle p-2">
            <div className="text-[10px] uppercase text-muted-foreground">{label}</div>
            <div className="font-mono text-[11px] font-bold">{value}</div>
          </div>
        ))}
      </div>

      <h4 className="field-label mt-5">Temperature history (backend)</h4>
      {loading ? (
        <p className="rounded-sm border border-border bg-subtle p-3 text-xs text-muted-foreground">
          Loading real sensor history…
        </p>
      ) : error ? (
        <p className="rounded-sm border border-destructive/40 bg-subtle p-3 text-xs text-destructive">{error}</p>
      ) : points.length ? (
        <>
          <div className="h-44 rounded-sm border border-border bg-surface py-2 pr-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points} margin={{ left: -18, top: 6 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" interval={2} />
                <YAxis tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" domain={["dataMin - 1", "dataMax + 1"]} unit="°" />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 2 }}
                  formatter={(value) => [`${Number(value).toFixed(1)}°C`, "Temperature"]}
                />
                {baseline !== null && (
                  <ReferenceLine
                    y={baseline}
                    stroke="var(--muted-foreground)"
                    strokeDasharray="4 4"
                    label={{ value: "Baseline", fontSize: 10, position: "insideTopLeft", fill: "var(--muted-foreground)" }}
                  />
                )}
                <Line type="monotone" dataKey="temp" stroke="var(--primary)" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            {[
              ["Baseline", baseline == null ? "—" : `${baseline.toFixed(1)}°C`],
              ["Current", current == null ? "—" : `${current.toFixed(1)}°C`],
              ["Deviation", deviation == null ? "—" : `${deviation >= 0 ? "+" : ""}${deviation.toFixed(1)}°C`],
            ].map(([label, value]) => (
              <div key={label} className="rounded-sm border border-border bg-subtle p-2">
                <div className="text-[10px] uppercase text-muted-foreground">{label}</div>
                <div className="font-mono text-sm font-bold">{value}</div>
              </div>
            ))}
          </div>

          <h4 className="field-label mt-5">Impact & deviation analysis</h4>
          {shifts.length ? (
            <ul className="space-y-2">
              {shifts.map((shift) => (
                <li
                  key={`${shift.time}-${shift.step}`}
                  className={`border-l-2 bg-subtle px-3 py-2 text-xs ${
                    shift.status === "SENSOR_ANOMALY" || Math.abs(shift.step) >= 6
                      ? "border-destructive"
                      : "border-sensor-warning"
                  }`}
                >
                  <p>{shift.text}</p>
                  <div className="mt-2 flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
                      <div
                        className={`h-full ${
                          shift.status === "SENSOR_ANOMALY" || Math.abs(shift.step) >= 6
                            ? "bg-destructive"
                            : "bg-sensor-warning"
                        }`}
                        style={{ width: `${shift.impact}%` }}
                      />
                    </div>
                    <span className="font-mono text-[11px] font-bold">{shift.impact}% deviation</span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-muted-foreground">
              No significant temperature movement is present in the backend history for this sensor.
            </p>
          )}
        </>
      ) : (
        <p className="rounded-sm border border-border bg-subtle p-3 text-xs text-muted-foreground">
          No historical telemetry has been stored for this sensor yet.
        </p>
      )}

      <p className="mt-4 text-[10px] text-muted-foreground">
        Last backend observation: {raw.observation_time || raw.timestamp || "unknown"}
      </p>
    </section>
  );
}
