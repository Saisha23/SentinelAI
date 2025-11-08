from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        # Store active WebSocket connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 Client connected: {len(self.active_connections)} active connections")

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"❌ Client disconnected: {len(self.active_connections)} active connections")

    async def send_message(self, message: str, websocket: WebSocket):
        """Send message to a specific client"""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            await connection.send_text(message)
