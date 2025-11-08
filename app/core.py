"""Central shared service instances to avoid circular imports.

Place commonly used singletons here (detection_service, connection_manager)
so other modules can import them without importing `app.main`.
"""
from app.services.connection_manager import ConnectionManager
from app.services.detection_service import DetectionService

# shared singletons
connection_manager = ConnectionManager()
detection_service = DetectionService()
