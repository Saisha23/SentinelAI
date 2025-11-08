// src/pages/Dashboard.jsx
import React, { useEffect, useState } from "react";
import { getZones, createZone, updateZone, deleteZone } from "../services/api";

const DashboardPage = () => {
  const [zones, setZones] = useState([]);
  const [form, setForm] = useState({ id: "", name: "", polygon: "" });
  const [editingId, setEditingId] = useState(null);

  // Load zones from backend
  useEffect(() => {
    fetchZones();
  }, []);

  const fetchZones = async () => {
    try {
      const data = await getZones();
      setZones(data);
    } catch (err) {
      console.error("Error fetching zones:", err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const polygon = form.polygon
      .split(";")
      .map((pair) => pair.trim().split(",").map(Number)); // "x1,y1; x2,y2"

    try {
      if (editingId) {
        await updateZone(editingId, { name: form.name, polygon });
      } else {
        await createZone({ ...form, polygon });
      }
      setForm({ id: "", name: "", polygon: "" });
      setEditingId(null);
      fetchZones();
    } catch (err) {
      console.error("Error saving zone:", err);
    }
  };

  const handleEdit = (zone) => {
    setForm({
      id: zone.id,
      name: zone.name,
      polygon: zone.polygon.map((p) => p.join(",")).join("; "),
    });
    setEditingId(zone.id);
  };

  const handleDelete = async (id) => {
    try {
      await deleteZone(id);
      fetchZones();
    } catch (err) {
      console.error("Error deleting zone:", err);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>📍 SentinelAI Zone Dashboard</h1>

      <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
        <input
          placeholder="Zone ID"
          value={form.id}
          onChange={(e) => setForm({ ...form, id: e.target.value })}
          disabled={!!editingId}
          required
          style={{ marginRight: "10px" }}
        />
        <input
          placeholder="Zone Name"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          required
          style={{ marginRight: "10px" }}
        />
        <input
          placeholder="Polygon (x1,y1; x2,y2; ...)"
          value={form.polygon}
          onChange={(e) => setForm({ ...form, polygon: e.target.value })}
          required
          style={{ marginRight: "10px", width: "300px" }}
        />
        <button type="submit">
          {editingId ? "Update Zone" : "Add Zone"}
        </button>
        {editingId && (
          <button
            type="button"
            onClick={() => {
              setForm({ id: "", name: "", polygon: "" });
              setEditingId(null);
            }}
            style={{ marginLeft: "10px" }}
          >
            Cancel
          </button>
        )}
      </form>

      <table border="1" cellPadding="8" style={{ borderRadius: "8px", margin: "0 auto" }}>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Polygon</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {zones.map((zone) => (
            <tr key={zone.id}>
              <td>{zone.id}</td>
              <td>{zone.name}</td>
              <td>{JSON.stringify(zone.polygon)}</td>
              <td>
                <button onClick={() => handleEdit(zone)}>Edit</button>
                <button onClick={() => handleDelete(zone.id)} style={{ marginLeft: "10px" }}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
          {zones.length === 0 && (
            <tr>
              <td colSpan="4">No zones found.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default DashboardPage;
