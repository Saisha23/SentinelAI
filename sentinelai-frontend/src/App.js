import { useEffect, useRef, useState } from "react";

function App() {
  const videoRef = useRef(null);
  const [detection, setDetection] = useState(null);
  const [alert, setAlert] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const streamSocket = new WebSocket("ws://127.0.0.1:8000/ws/stream");
    const alertSocket = new WebSocket("ws://127.0.0.1:8000/ws/alerts");

    // Handle open/close
    streamSocket.onopen = () => {
      console.log("✅ Connected to stream socket");
      setConnected(true);
      startCamera(streamSocket);
    };
    streamSocket.onclose = () => {
      console.warn("⚠️ Stream socket closed");
      setConnected(false);
    };

    alertSocket.onopen = () => console.log("✅ Connected to alert socket");
    alertSocket.onclose = () => console.warn("⚠️ Alert socket closed");

    // Handle incoming data
    streamSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetection(data);
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

  const startCamera = (streamSocket) => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          const video = videoRef.current;

          const sendFrame = () => {
            if (!connected || video.readyState !== 4) return;

            const canvas = document.createElement("canvas");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL("image/jpeg").split(",")[1]; // Base64
            streamSocket.send(frameData);
          };

          setInterval(sendFrame, 300); // Send every 300ms
        }
      })
      .catch((err) => console.error("Camera access denied:", err));
  };

  return (
    <div style={{ textAlign: "center", marginTop: "30px" }}>
      <h1>🎥 SentinelAI Live Detection</h1>
      <video ref={videoRef} autoPlay playsInline style={{ width: "60%", borderRadius: "10px" }} />
      <div style={{ marginTop: "20px" }}>
        {detection ? (
          <pre>{JSON.stringify(detection, null, 2)}</pre>
        ) : (
          <p>Waiting for detection data...</p>
        )}
        {alert && <p style={{ color: "red" }}>🚨 Alert: {alert.message || "Threat detected!"}</p>}
      </div>
    </div>
  );
}

export default App;
