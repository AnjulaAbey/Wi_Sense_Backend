from channels.generic.websocket import AsyncWebsocketConsumer
import json
from cms.models import CSIData
from channels.db import database_sync_to_async

class HealthCheckConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(json.dumps({'status': 'OK', 'message': 'WebSocket is working correctly!'}))

    async def disconnect(self, close_code):
        pass

class CSIDataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    @database_sync_to_async
    def save_csi_data(self, data):
        CSIData.objects.create(
            type=data["type"],
            mac=data["mac"],
            rssi=data["rssi"],
            time_stamp=data["time_stamp"],
            raw_data=data["csi_raw_data"],
            port_time_stamp=data["port_time_stamp"]
            # other fields
        )

    async def receive(self, text_data):
        # process the incoming message
        data = json.loads(text_data)
        
        # Call the async version of the ORM call
        await self.save_csi_data(data)