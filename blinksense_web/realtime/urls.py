"""
URL configuration for realtime app
"""
from django.urls import path
from . import views
from . import views_simple

urlpatterns = [
    path('', views.index, name='index'),
    path('api/process-frame/', views_simple.process_frame, name='process_frame'),
    path('api/health/', views_simple.health_check, name='health_check'),
]