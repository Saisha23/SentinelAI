import React, { useEffect } from "react";

function TestConnection() {
  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8000/ws/stream");

    socket.onopen = () => {
      console.log("✅ Connected to FastAPI backend");
      socket.send("Hello from frontend!");
    };

    socket.onmessage = (event) => {
      console.log("📩 Message from backend:", event.data);
    };

    socket.onerror = (error) => {
      console.error("❌ WebSocket error:", error);
    };

    socket.onclose = () => {
      console.log("🔌 Connection closed");
    };

    return () => socket.close();
  }, []);

  return <h2>WebSocket Connection Test</h2>;
}

export default TestConnection;
