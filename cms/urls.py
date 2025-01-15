from rest_framework import routers
from .views import CSIDataViewSet
from django.urls import path, include

router = routers.DefaultRouter()
router.register('csidata', CSIDataViewSet)
urlpatterns = [
    path('', include(router.urls)),
]