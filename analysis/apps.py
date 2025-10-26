# analysis/apps.py

from django.apps import AppConfig

# Importa la función de carga que está en analysis_script.py (asumo que está en el mismo directorio)
# Si analysis_script está en la misma carpeta 'analysis', la importación debe ser relativa.
from .analysis_script import load_global_resources 

class AnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analysis' 

    def ready(self):
        """
        Llama a la función de carga de recursos de ML al iniciar el servidor
        para que los modelos estén disponibles globalmente.
        """
        print("--- INICIO: Preparando la carga de recursos de ML ---")
        load_global_resources()
        print("--- FIN: Intento de carga de recursos de ML completado ---")