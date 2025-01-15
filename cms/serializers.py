from rest_framework import serializers
from .models import CSIData

class CSIDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = CSIData
        fields = '__all__'
