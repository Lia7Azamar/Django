#!/usr/bin/env bash

# Espera 5 segundos para que los procesos de inicialización interna de Python/Gunicorn finalicen.
sleep 5

# Inicia Gunicorn, enlazando al puerto que Render proporciona
gunicorn malware_project.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 60 --log-file -