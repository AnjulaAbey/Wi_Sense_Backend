from rest_framework import routers
from .views import CSIDataViewSet, RealTimePresenceDetection, RealTimePostureDetection
from django.urls import path, include

router = routers.DefaultRouter()
router.register('csidata', CSIDataViewSet)
router.register('presence', RealTimePresenceDetection, basename='realtime-prediction')
router.register('posture', RealTimePostureDetection, basename='realtime-posture-prediction')
urlpatterns = [
    path('', include(router.urls)),
]