=======================================================================
               GUÍA DE ARRANQUE DEL PROYECTO
=======================================================================

Este proyecto consta de dos partes independientes:
1. Backend (FastAPI / Python)
2. Frontend (React / Vite)

Para probar la aplicación completa, debes arrancar AMBOS servicios
en terminales separadas.

-----------------------------------------------------------------------
OPCIÓN A: EJECUCIÓN MANUAL EN PARALELO (Recomendado para desarrollo)
-----------------------------------------------------------------------

TERMINAL 1: BACKEND (Servidor API)
1. Abre una terminal y colócate en la raíz del proyecto.
2. Entra a la carpeta del backend:
   cd backend
3. Asegúrate de activar tu entorno virtual (.venv):
   source .venv/bin/activate
4. Ejecuta la variable de entorno para las rutas de Python y arranca FastAPI:
   export PYTHONPATH=$PYTHONPATH:.
   fastapi dev src/api/main.py

* El backend estará listo en: http://127.0.0.1:8000
* La documentación interactiva (Swagger): http://127.0.0.1:8000/docs


TERMINAL 2: FRONTEND (Interfaz Web)
1. Abre una SEGUNDA terminal y colócate en la raíz del proyecto.
2. Entra a la carpeta del frontend:
   cd frontend
3. (Solo la primera vez) Instala las dependencias si no lo has hecho:
   npm install
4. Arranca el servidor de Vite:
   npm run dev

* La interfaz web estará lista en: http://localhost:5173/

=======================================================================