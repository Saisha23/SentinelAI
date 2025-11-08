// src/services/api.js
const API_BASE = "http://127.0.0.1:8000/api/zones"; // adjust if your backend runs elsewhere

// Fetch all zones
export async function getZones() {
  const res = await fetch(`${API_BASE}/zones`);
  if (!res.ok) throw new Error("Failed to fetch zones");
  return await res.json();
}

// Create a new zone
export async function createZone(zone) {
  const res = await fetch(`${API_BASE}/zones`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(zone),
  });
  if (!res.ok) throw new Error("Failed to create zone");
  return await res.json();
}

// Update an existing zone
export async function updateZone(id, data) {
  const res = await fetch(`${API_BASE}/zones/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to update zone");
  return await res.json();
}

// Delete a zone
export async function deleteZone(id) {
  const res = await fetch(`${API_BASE}/zones/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete zone");
  return await res.json();
}
