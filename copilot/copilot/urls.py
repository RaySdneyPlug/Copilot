from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('copilot_app.urls')),  # Inclui as URLs do app principal
]
