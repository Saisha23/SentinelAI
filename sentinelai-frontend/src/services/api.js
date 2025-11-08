import { useEffect, useState } from "react";
import { streamSocket, alertSocket } from "./services/socket";

function App() {
  const [frame, setFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    streamSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.frame) setFrame(data.frame);
      if (data.detections) setDetections(data.detections);
    };

    alertSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setAlert(data);
    };

    return () => {
      streamSocket.close();
      alertSocket.close();
    };
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "30px" }}>
      <h1>🎯 SentinelAI — Live Video Stream</h1>

      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt="Live Stream"
          style={{ width: "70%", borderRadius: "10px", boxShadow: "0 0 15px rgba(0,0,0,0.3)" }}
        />
      ) : (
        <p>Waiting for live video feed...</p>
      )}

      {detections.length > 0 && (
        <p style={{ color: "green" }}>
          ✅ Objects detected: {detections.join(", ")}
        </p>
      )}

      {alert && (
        <p style={{ color: "red", fontWeight: "bold" }}>
          🚨 Alert: {alert.message || "Suspicious activity detected!"}
        </p>
      )}
    </div>
  );
}

export default App;
