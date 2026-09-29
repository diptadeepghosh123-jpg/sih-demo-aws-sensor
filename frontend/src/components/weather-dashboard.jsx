import { Link } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Marker,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  MapPin,
  Menu,
  MoreVertical,
  Plus,
  Radio,
  Satellite,
  Trash2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import SensorDetails from "@/components/sensor-details";
import SkyGuardChatbot from "@/components/skyguard-chatbot";
import { addSensor, deleteSensor, getSensors } from "@/api";

const INDIA_CENTER = [23.8, 78.5];
const REFRESH_MS = 30000;

function canonicalSensorId(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/[\s_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function mapBackendSensor(item) {
  const status = String(item.status || "NORMAL").toUpperCase();
  const state =
    status === "SENSOR_ANOMALY" || status === "GLITCH_SUSPECT"
      ? "anomaly"
      : status === "WATCH" || status === "WEATHER_EVENT" || status === "WEATHER EVENT"
        ? "warning"
        : status === "DATA_UNAVAILABLE" || status === "API_TIMEOUT" || status === "API_ERROR"
          ? "warning"
          : "normal";

  const temperature = item.temperature == null ? "—" : `${Number(item.temperature).toFixed(1)}°C`;
  const pressure = item.pressure == null ? "—" : `${Number(item.pressure).toFixed(1)} hPa`;
  const humidity = item.humidity == null ? "—" : `${Number(item.humidity).toFixed(1)}% RH`;

  return {
    id: String(item.sensor_id),
    name: item.locality || item.city || item.district || String(item.sensor_id),
    lat: Number(item.latitude),
    lng: Number(item.longitude),
    state,
    reading: `${temperature} · ${pressure} · ${humidity}`,
    updated: item.observation_time || item.timestamp || "",
    raw: item,
  };
}

function messageForSensor(sensor) {
  const raw = sensor.raw || {};
  if (raw.explanation) return raw.explanation;
  if (raw.fault_type && raw.fault_type !== "None") return raw.fault_type;
  if (raw.anomaly_category) return raw.anomaly_category.replaceAll("_", " ");
  return "Backend flagged this station for review.";
}

function warningFromSensor(sensor) {
  const status = String(sensor.raw?.status || "").toUpperCase();
  return {
    sensorId: sensor.id,
    type: status === "WEATHER_EVENT" || status === "WEATHER EVENT" ? "Possible Weather Event" : "Watch",
    message: messageForSensor(sensor),
    time: sensor.updated,
  };
}

function criticalFromSensor(sensor) {
  return {
    sensorId: sensor.id,
    type: sensor.raw?.fault_type && sensor.raw.fault_type !== "None" ? sensor.raw.fault_type : "Sensor anomaly",
    message: messageForSensor(sensor),
    time: sensor.updated,
  };
}

function ViewController({ focus, visibleSensors }) {
  const map = useMap();

  useEffect(() => {
    if (focus) {
      map.flyTo([focus.lat, focus.lng], 8, { duration: 0.8 });
    }
  }, [focus, map]);

  useEffect(() => {
    if (!focus && visibleSensors.length > 1) {
      const bounds = L.latLngBounds(visibleSensors.map((sensor) => [sensor.lat, sensor.lng]));
      if (bounds.isValid()) map.fitBounds(bounds.pad(0.15), { maxZoom: 10, animate: false });
    }
  }, [focus, visibleSensors, map]);

  return null;
}

function MapInteraction({ placement, deletion, onPlace, onDelete }) {
  useMapEvents({
    click(event) {
      if (placement) onPlace({ lat: event.latlng.lat, lng: event.latlng.lng });
      if (deletion) onDelete();
    },
  });
  return null;
}

const testIcon = L.divIcon({
  className: "test-sensor-icon",
  html: "<span><i></i></span>",
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

export default function WeatherDashboard() {
  const [sensors, setSensors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");
  const [panelOpen, setPanelOpen] = useState(true);
  const [focus, setFocus] = useState(null);
  const [placement, setPlacement] = useState(false);
  const [deletion, setDeletion] = useState(false);
  const [draft, setDraft] = useState(null);
  const [sensorId, setSensorId] = useState("");
  const [layer, setLayer] = useState("road");
  const [selectedState, setSelectedState] = useState("ALL");
  const [notice, setNotice] = useState("");
  const mapRef = useRef(null);

  const loadSensors = async () => {
    try {
      setApiError("");
      const data = await getSensors();
      const mapped = Array.isArray(data?.sensors) ? data.sensors.map(mapBackendSensor) : [];
      setSensors(mapped);
      setLastUpdated(new Date());
    } catch (error) {
      console.error(error);
      setApiError("Backend unavailable — start FastAPI on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSensors();
    const timer = window.setInterval(loadSensors, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, []);

  const states = useMemo(
    () => Array.from(new Set(sensors.map((sensor) => sensor.raw?.state).filter(Boolean))).sort(),
    [sensors],
  );

  const visibleSensors = useMemo(
    () => selectedState === "ALL" ? sensors : sensors.filter((sensor) => sensor.raw?.state === selectedState),
    [sensors, selectedState],
  );

  const anomalySensors = visibleSensors.filter((sensor) => sensor.state === "anomaly");
  const warningSensors = visibleSensors.filter((sensor) => sensor.state === "warning");
  const activeCount = visibleSensors.filter((sensor) => sensor.state === "normal").length;
  const warnings = warningSensors.map(warningFromSensor);
  const criticals = anomalySensors.map(criticalFromSensor);

  const tileUrl =
    layer === "road"
      ? "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      : "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png";

  const sensorColor = (state) =>
    state === "anomaly"
      ? "var(--sensor-critical)"
      : state === "warning"
        ? "var(--sensor-warning)"
        : "var(--sensor-normal)";

  const selectSensor = (id) => {
    const sensor = sensors.find((item) => item.id === id);
    if (sensor) {
      setFocus({ ...sensor });
      setSelectedState(sensor.raw?.state || "ALL");
    }
  };

  const saveSensor = async () => {
    const normalized = canonicalSensorId(sensorId);
    if (!draft || normalized.length < 3 || normalized.length > 30 || !/^[A-Z0-9_]+$/.test(normalized)) return;
    if (sensors.some((sensor) => sensor.id === normalized)) {
      setNotice("Sensor ID already exists");
      return;
    }

    const next = {
      id: normalized,
      name: "New field station",
      ...draft,
      state: "normal",
      reading: "Awaiting backend telemetry",
      updated: "Pending",
      raw: {
        sensor_id: normalized,
        latitude: draft.lat,
        longitude: draft.lng,
        state: selectedState === "ALL" ? "Unassigned" : selectedState,
        status: "NORMAL",
        temperature: null,
        pressure: null,
        humidity: null,
      },
    };

    try {
      await addSensor({
        sensor_id: normalized,
        latitude: draft.lat,
        longitude: draft.lng,
        state: selectedState === "ALL" ? "Unassigned" : selectedState,
        district: "",
        city: "",
        locality: "New field station",
      });
      setFocus(next);
      setDraft(null);
      setPlacement(false);
      setSensorId("");
      setNotice(`${normalized} added to the backend sensor registry`);
      await loadSensors();
    } catch (error) {
      console.error(error);
      setNotice(`Could not add ${normalized}: ${error.message}`);
    }
    window.setTimeout(() => setNotice(""), 5000);
  };

  const deleteFocused = async () => {
    if (!focus) {
      setDeletion(true);
      setNotice("Select a sensor dot to delete");
      return;
    }
    try {
      await deleteSensor(focus.id);
      setNotice(`${focus.id} removed from the backend sensor registry`);
      setFocus(null);
      setDeletion(false);
      await loadSensors();
    } catch (error) {
      console.error(error);
      setNotice(`Could not delete ${focus.id}: ${error.message}`);
    }
    window.setTimeout(() => setNotice(""), 2500);
  };

  const coordinates = useMemo(
    () => (draft ? `${draft.lat.toFixed(5)}, ${draft.lng.toFixed(5)}` : ""),
    [draft],
  );

  const updatedLabel = lastUpdated
    ? lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  return (
    <main className="h-dvh w-full overflow-hidden bg-background text-foreground">
      <header className="absolute inset-x-0 top-0 z-[1000] flex h-16 items-center justify-between border-b border-border bg-surface/95 px-4 shadow-chrome backdrop-blur-sm md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-sm bg-primary text-primary-foreground">
            <Satellite className="size-5" />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-bold uppercase tracking-wide md:text-base">SkyGuard AI</h1>
            <p className="truncate text-xs text-muted-foreground">Real-Time Weather Anomaly Detection</p>
          </div>
        </div>

        <div className="hidden items-center gap-6 md:flex">
          <div className="stat"><span className="status-dot bg-sensor-normal" /><div><b>{activeCount}</b><small>Normal</small></div></div>
          <div className="stat"><span className="status-dot bg-sensor-warning" /><div><b>{warningSensors.length}</b><small>Warnings</small></div></div>
          <div className="stat"><span className="status-dot bg-sensor-critical" /><div><b>{anomalySensors.length}</b><small>Critical</small></div></div>
          <div className="border-l border-border pl-6 text-right">
            <p className={`text-xs font-semibold ${apiError ? "text-destructive" : ""}`}>{apiError ? "NETWORK OFFLINE" : "NETWORK LIVE"}</p>
            <p className="font-mono text-[11px] text-muted-foreground">UPDATED {updatedLabel}</p>
          </div>
        </div>

        <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setPanelOpen(!panelOpen)} aria-label="Toggle alerts panel">
          <Menu />
        </Button>
      </header>

      <section className="absolute inset-x-0 bottom-0 top-16">
        <MapContainer
          ref={mapRef}
          center={INDIA_CENTER}
          zoom={5}
          minZoom={4}
          maxZoom={15}
          zoomControl={false}
          className="h-full w-full"
          attributionControl
        >
          <TileLayer
            key={layer}
            url={tileUrl}
            attribution={layer === "road" ? "&copy; OpenStreetMap contributors" : "Map data &copy; OpenStreetMap contributors, SRTM | Map style &copy; OpenTopoMap"}
          />
          <ViewController focus={focus} visibleSensors={visibleSensors} />
          <MapInteraction
            placement={placement}
            deletion={deletion}
            onPlace={(point) => { setDraft(point); setPlacement(false); setDeletion(false); }}
            onDelete={() => setDeletion(false)}
          />

          {visibleSensors.map((sensor) => (
            <CircleMarker
              key={sensor.id}
              center={[sensor.lat, sensor.lng]}
              radius={focus?.id === sensor.id ? 10 : 7}
              pathOptions={{
                color: "var(--map-marker-ring)",
                weight: focus?.id === sensor.id ? 4 : 2,
                fillColor: sensorColor(sensor.state),
                fillOpacity: 1,
              }}
              eventHandlers={{
                click: () => deletion
                  ? (setSensors((all) => all.filter((s) => s.id !== sensor.id)), setDeletion(false), setNotice(`${sensor.id} removed from this dashboard view`))
                  : setFocus({ ...sensor }),
              }}
            >
              <Tooltip direction="top" offset={[0, -8]}>
                <div className="sensor-tooltip">
                  <strong>{sensor.id}</strong>
                  <span>{sensor.name}</span>
                  <span>{sensor.reading}</span>
                  <span>{sensor.raw?.status || "NORMAL"}</span>
                </div>
              </Tooltip>
            </CircleMarker>
          ))}

          {draft && (
            <Marker
              draggable
              position={[draft.lat, draft.lng]}
              icon={testIcon}
              eventHandlers={{
                dragend: (event) => {
                  const point = event.target.getLatLng();
                  setDraft({ lat: point.lat, lng: point.lng });
                },
              }}
            />
          )}
        </MapContainer>

        <div className="absolute left-4 top-4 z-[500] flex flex-wrap overflow-hidden rounded-sm border border-border bg-surface shadow-chrome">
          <Button size="sm" variant={layer === "road" ? "default" : "ghost"} className="rounded-none shadow-none" onClick={() => setLayer("road")}>Road</Button>
          <Button size="sm" variant={layer === "terrain" ? "default" : "ghost"} className="rounded-none shadow-none" onClick={() => setLayer("terrain")}>Terrain</Button>
          <select
            value={selectedState}
            onChange={(event) => { setSelectedState(event.target.value); setFocus(null); }}
            className="h-9 border-l border-border bg-surface px-2 text-xs font-medium outline-none"
            aria-label="Filter by state"
          >
            <option value="ALL">All states</option>
            {states.map((state) => <option key={state} value={state}>{state}</option>)}
          </select>
        </div>

        <div className="absolute bottom-7 left-4 z-[500] flex flex-col overflow-hidden rounded-sm border border-border bg-surface shadow-chrome">
          <Button variant="ghost" size="icon" className="rounded-none border-b border-border" onClick={() => mapRef.current?.zoomIn()} aria-label="Zoom in"><Plus /></Button>
          <Button variant="ghost" size="icon" className="rounded-none" onClick={() => mapRef.current?.setView(INDIA_CENTER, 5)} aria-label="Reset map view"><Crosshair /></Button>
        </div>

        {panelOpen ? (
          <aside className="absolute bottom-4 right-4 top-4 z-[600] flex w-[calc(100%-2rem)] flex-col overflow-hidden rounded-sm border border-border bg-surface shadow-panel sm:w-[390px]">
            <div className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4">
              <div className="flex items-center gap-2"><Radio className="size-4 text-primary" /><h2 className="text-sm font-bold uppercase tracking-wide">Live Operations</h2></div>
              <div className="flex items-center">
                <Link to="/history" aria-label="Open anomaly history dashboard" title="Anomaly history" className="inline-flex size-9 items-center justify-center rounded-md hover:bg-accent"><MoreVertical className="size-4" /></Link>
                <Button variant="ghost" size="icon" onClick={() => setPanelOpen(false)} aria-label="Collapse side panel"><ChevronRight /></Button>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto">
              {loading && !sensors.length ? (
                <p className="p-4 text-xs text-muted-foreground">Loading live station network…</p>
              ) : focus ? (
                <SensorDetails key={focus.id} sensor={focus} onBack={() => setFocus(null)} />
              ) : (
                <>
                  {apiError && <p className="border-b border-destructive/30 bg-destructive/5 px-4 py-3 text-xs text-destructive">{apiError}</p>}

                  <section className="border-b border-border">
                    <div className="section-label critical-label"><span>Critical sensor anomalies</span><b>{criticals.length}</b></div>
                    <div className="p-3">
                      {criticals.length ? criticals.map((warning) => (
                        <button key={`${warning.sensorId}-${warning.type}`} className="alert-row critical-row" onClick={() => selectSensor(warning.sensorId)}>
                          <span className="alert-icon critical-icon"><AlertTriangle /></span>
                          <span className="min-w-0 flex-1">
                            <span className="alert-title"><b>{warning.sensorId}</b><time>{warning.time}</time></span>
                            <span className="alert-place">{warning.type}</span>
                            <span className="alert-message">{warning.message}</span>
                          </span>
                          <MapPin className="size-4 shrink-0 text-muted-foreground" />
                        </button>
                      )) : <p className="px-3 pb-3 text-xs text-muted-foreground">No critical sensor anomalies in the selected state.</p>}
                    </div>
                  </section>

                  <section>
                    <div className="section-label"><span>Warnings & events</span><b>{warnings.length}</b></div>
                    <div className="divide-y divide-border px-3">
                      {warnings.length ? warnings.map((warning) => (
                        <button key={`${warning.sensorId}-${warning.type}`} className="alert-row" onClick={() => selectSensor(warning.sensorId)}>
                          <span className="alert-icon warning-icon"><AlertTriangle /></span>
                          <span className="min-w-0 flex-1">
                            <span className="alert-title"><b>{warning.type}</b><time>{warning.time}</time></span>
                            <span className="alert-place">{warning.sensorId}</span>
                            <span className="alert-message">{warning.message}</span>
                          </span>
                          <MapPin className="size-4 shrink-0 text-muted-foreground" />
                        </button>
                      )) : <p className="px-3 py-3 text-xs text-muted-foreground">No warnings or weather events in the selected state.</p>}
                    </div>
                  </section>
                </>
              )}
            </div>

            <section className="shrink-0 border-t border-border bg-subtle p-4">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wide">Sensor management</h3>
                <span className="font-mono text-[11px] text-muted-foreground">{visibleSensors.length} STATIONS</span>
              </div>

              {draft && (
                <div className="mb-3 space-y-3 rounded-sm border border-primary/30 bg-surface p-3">
                  <div className="flex items-center justify-between"><p className="text-xs font-semibold">New sensor location</p><Button variant="ghost" size="icon" className="size-7" onClick={() => setDraft(null)} aria-label="Cancel adding sensor"><X /></Button></div>
                  <div><label className="field-label" htmlFor="coordinates">Coordinates</label><Input id="coordinates" value={coordinates} readOnly className="font-mono text-xs" /></div>
                  <div><label className="field-label" htmlFor="sensor-id">Sensor ID</label><Input id="sensor-id" value={sensorId} maxLength={30} placeholder="AWS-XXX-000" onChange={(event) => { setSensorId(event.target.value); setNotice(""); }} onKeyDown={(event) => event.key === "Enter" && saveSensor()} aria-invalid={Boolean(notice && notice.includes("exists"))} /></div>
                  <Button className="w-full" onClick={saveSensor} disabled={!/^[A-Za-z0-9_-]{3,30}$/.test(sensorId.trim())}>Add to network</Button>
                </div>
              )}

              {placement && <p className="mb-3 border-l-2 border-primary bg-surface px-3 py-2 text-xs font-medium">Click anywhere on the map to place the sensor.</p>}
              {deletion && <p className="mb-3 border-l-2 border-destructive bg-surface px-3 py-2 text-xs font-medium">Select a sensor dot to remove it.</p>}
              {notice && <p className="mb-3 text-xs font-medium text-primary" role="status">{notice}</p>}

              <div className="grid grid-cols-2 gap-2">
                <Button onClick={() => { setPlacement(true); setDeletion(false); setDraft(null); }}><Plus />Add Sensor</Button>
                <Button variant="outline" onClick={deleteFocused}><Trash2 />Delete Sensor</Button>
              </div>
            </section>
          </aside>
        ) : (
          <Button className="absolute right-4 top-4 z-[600] shadow-panel" size="icon" onClick={() => setPanelOpen(true)} aria-label="Open alerts panel"><ChevronLeft /></Button>
        )}
      </section>
      <SkyGuardChatbot sensorId={focus?.id || null} />
    </main>
  );
}
