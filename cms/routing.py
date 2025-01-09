from django.urls import re_path
from .consumers import HealthCheckConsumer , CSIDataConsumer

websocket_urlpatterns = [
    re_path(r'ws/health/$', HealthCheckConsumer.as_asgi()),
    re_path("ws/csi/", CSIDataConsumer.as_asgi()),
]
