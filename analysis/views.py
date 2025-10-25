# views.py (No requiere cambios)

from django.http import JsonResponse
from rest_framework.decorators import api_view
# Importación relativa: importa la función del archivo analysis_script.py
from .analysis_script import run_malware_analysis 

@api_view(['GET'])
def api_malware_results(request):
    """
    Ejecuta el script de Machine Learning y devuelve los resultados en formato JSON.
    """
    try:
        # Ejecuta la función principal de análisis
        results = run_malware_analysis()
        
        # Devolver el diccionario de resultados como una respuesta JSON
        return JsonResponse(results, status=200, safe=False)

    except FileNotFoundError:
        return JsonResponse({
            'error': 'FileNotFoundError',
            'message': 'El archivo TotalFeatures-ISCXFlowMeter.csv no se encontró. Asegúrate de que esté en la misma carpeta que analysis_script.py.'
        }, status=500)
    except Exception as e:
        return JsonResponse({
            'error': 'InternalServerError',
            'message': f'Error al ejecutar ML: {str(e)}' 
        }, status=500)