const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options.timeout ?? 20000);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload?.detail || detail;
      } catch {
        // Keep the HTTP status when the server did not return JSON.
      }
      throw new Error(`API ${response.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }

    return await response.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

export function getSensors() {
  return request("/api/analyze", { timeout: 60000 });
}

export function getSensorRegistry() {
  return request("/api/sensors");
}

export function getSensorSummary() {
  return request("/api/sensors/summary");
}

export function getSensorStates() {
  return request("/api/sensors/states");
}

export function addSensor(sensor) {
  return request("/api/sensors", {
    method: "POST",
    body: JSON.stringify(sensor),
  });
}

export function updateSensor(sensorId, sensor) {
  return request(`/api/sensors/${encodeURIComponent(sensorId)}`, {
    method: "PUT",
    body: JSON.stringify(sensor),
  });
}

export function deleteSensor(sensorId) {
  return request(`/api/sensors/${encodeURIComponent(sensorId)}`, {
    method: "DELETE",
  });
}

export function getHistory() {
  return request("/api/history", { timeout: 20000 });
}

export function getMaintenance(sensorId) {
  return request(`/api/maintenance/${encodeURIComponent(sensorId)}`);
}

export function testAnomaly(type, sensorId) {
  const query = sensorId ? `?sensor_id=${encodeURIComponent(sensorId)}` : "";
  return request(`/api/test/${encodeURIComponent(type)}${query}`, { timeout: 60000 });
}

export function sendChatMessage(message, sensorId = null) {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, sensor_id: sensorId }),
    timeout: 20000,
  });
}

export { API_BASE };
