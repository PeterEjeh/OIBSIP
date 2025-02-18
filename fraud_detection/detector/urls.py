import django.urls
from . import views

urlpatterns = [
    django.urls.path('predict/', views.predict_fraud, name='predict_fraud'),
]