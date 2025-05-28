from django.db import models

# Create your models here.
class CSIData(models.Model):
    type = models.CharField(max_length=255)
    mac = models.CharField(max_length=255)
    rssi = models.FloatField()
    time_stamp = models.CharField(max_length=255)
    raw_data = models.JSONField()
    # Add other fields as needed
    created_at = models.DateTimeField(auto_now_add=True)
    port_time_stamp = models.CharField(max_length=255, null=True, blank=True)
