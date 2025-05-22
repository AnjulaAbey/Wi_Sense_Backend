from rest_framework import routers
from .views import CSIDataViewSet, RealTimePresenceDetection
from django.urls import path, include

router = routers.DefaultRouter()
router.register('csidata', CSIDataViewSet)
router.register('presence', RealTimePresenceDetection, basename='realtime-prediction')
urlpatterns = [
    path('', include(router.urls)),
]