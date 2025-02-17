from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StockPredictorViewSet

router = DefaultRouter()
router.register(r'predictor', StockPredictorViewSet, basename='predictor')

urlpatterns = [
    path('', include(router.urls)),
]