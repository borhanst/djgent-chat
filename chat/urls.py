"""
URL configuration for the chat application.
"""

from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    # Main views
    path('', views.chat_view, name='chat'),
    path('settings/', views.settings_view, name='settings'),
    
    # API endpoints
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/conversation/<str:session_id>/', views.conversation_api, name='conversation_api'),
    path('api/index-status/', views.index_status_api, name='index_status_api'),
    path('api/new-session/', views.new_session_api, name='new_session_api'),
]
