# Proyecto I
# Predicción de Tratamiento para Pacientes Vasculares

## Misión del Proyecto
Se pretende conseguir un modelo que pueda predecir si un paciente recibirá tratamiento endovascular (EVAR, Endovascular Aneurysm Repair) (2), cirugía abierta (1) o manejo conservador (0)
Esta predicción se realiza basándose en las siguientes variables de entrada: edad, sexo, fumador, comorbilidades (HTA, IAM, ERC, EPOC, ACV, FA, insuficiencia cardíaca).

El propósito final es desarrollar la base del software que provea un servicio a médicos para que el diagnóstico de un paciente se pueda hacer con mayor seguridad

## Objetivos
Precisión y Tiempo: Entrenar un modelo (iniciando con regresión logística y explorando redes neuronales) con los datos reales de clínicas en un plazo de 3 meses que describa con un 75% de precisión el tipo de tratamiento adecuado.
Interactividad: Crear una interfaz interactiva para el usuario donde pueda cargar datos de un paciente, o un dataset entero y nuestro modelo devuelva un diagnóstico adecuado
Explicabilidad: Obtener un modelo que pueda explicar el motivo por el que devuelve cierto diagnóstico

## Arquitectura y Estructura del Repositorio

El proyecto sigue una arquitectura modular para separar la limpieza de datos, el entrenamiento del modelo y la interfaz gráfica.

text
proyecto_vascular/
│
├── data/                   # Carpeta para los datasets (ignorado en .gitignore)
│   ├── raw/                # Datos originales de la clínica
│   └── processed/          # Datos limpios y listos para entrenar
│
├── docs/                   # Documentación y entregables
│   ├── memoria_proyecto.pdf# Memoria detallada del planteamiento, métodos y resultados
│   └── presentacion.pdf    # Presentación visual del proyecto
│
├── notebooks/              # Jupyter Notebooks para exploración (EDA)
│   └── 01_exploracion_y_limpieza.ipynb
│
├── src/                    # Código fuente principal (Módulos de Python)
│   ├── __init__.py
│   ├── data_prepro.py      # Lógica de limpieza y manejo de datos faltantes
│   ├── train_model.py      # Script para entrenar la regresión logística / red neuronal
│   └── explainability.py   # Código para la explicabilidad del modelo (SHAP, LIME, etc.)
│
├── app/                    # Código de la interfaz de usuario interactiva
│   └── main_app.py         # Aplicación (ej. Streamlit o Gradio)
│
├── README.md               # Este archivo
├── requirements.txt        # Dependencias y librerías necesarias
└── video_explicativo.url   # Enlace al video corto explicando el proyecto
