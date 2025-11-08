import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import App from "./App";
import DashboardPage from "./pages/DashboardPage";

const Root = () => (
  <Router>
    <nav style={{ padding: "10px", background: "#eee", marginBottom: "15px" }}>
      <Link to="/" style={{ marginRight: "10px" }}>🎥 Live Detection</Link>
      <Link to="/dashboard">📊 Dashboard</Link>
    </nav>

    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/dashboard" element={<DashboardPage />} />
    </Routes>
  </Router>
);

ReactDOM.createRoot(document.getElementById("root")).render(<Root />);
