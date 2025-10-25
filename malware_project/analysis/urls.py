from django.urls import path

# Importa las vistas de la aplicación actual (analysis/views.py)
from . import views 

urlpatterns = [
    # Esta es la ruta /api/malware/results/ (usando el prefijo definido en el urls.py principal)
    path('results/', views.api_malware_results, name='malware_results'),
]
