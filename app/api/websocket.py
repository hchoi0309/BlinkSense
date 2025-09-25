"""
FastAPI WebSocket endpoint for drowsiness detection
Handles real-time video frame processing
"""
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect
from ..models import DrowsinessDetector

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept WebSocket connection and create detector instance"""
        await websocket.accept()
        if client_id is None:
            client_id = id(websocket)

        # Create a new detector instance for each connection
        detector = DrowsinessDetector()
        self.active_connections[client_id] = {
            "websocket": websocket,
            "detector": detector
        }

        logger.info(f"WebSocket connected: {client_id}")
        return client_id

    def disconnect(self, client_id: str):
        """Remove connection and clean up detector"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_message(self, client_id: str, message: dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send_text(json.dumps(message))

    def get_detector(self, client_id: str) -> DrowsinessDetector:
        """Get detector instance for client"""
        if client_id in self.active_connections:
            return self.active_connections[client_id]["detector"]
        return None


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for drowsiness detection
    Compatible with existing BlinkSense client protocol
    """
    client_id = None

    try:
        # Accept connection and create detector
        client_id = await manager.connect(websocket)

        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "frame":
                    # Process video frame
                    frame_data = message.get("frame")
                    if frame_data:
                        detector = manager.get_detector(client_id)
                        if detector:
                            result = detector.process_frame(frame_data)
                            await manager.send_message(client_id, result)
                        else:
                            await manager.send_message(client_id, {
                                "status": "error",
                                "message": "Detector not initialized"
                            })
                    else:
                        await manager.send_message(client_id, {
                            "status": "error",
                            "message": "No frame data received"
                        })
                else:
                    # Unknown message type
                    await manager.send_message(client_id, {
                        "status": "error",
                        "message": f"Unknown message type: {message.get('type')}"
                    })

            except json.JSONDecodeError:
                await manager.send_message(client_id, {
                    "status": "error",
                    "message": "Invalid JSON format"
                })

            except Exception as e:
                logger.error(f"Error processing message from {client_id}: {e}")
                await manager.send_message(client_id, {
                    "status": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")

    finally:
        # Clean up connection
        if client_id:
            manager.disconnect(client_id)