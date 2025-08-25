"""
WebSocket URL routing for realtime drowsiness detection
"""
from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/drowsiness/', consumers.DrowsinessConsumer.as_asgi()),
]