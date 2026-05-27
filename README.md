# Proyecto I: Predicción de Tratamiento para Pacientes Vasculares

## Misión del Proyecto
El propósito de este proyecto es proveer un sistema integral (software y modelos predictivos) que asista a los médicos en el diagnóstico y decisión de tratamiento para pacientes vasculares. El sistema evalúa datos del paciente (edad, sexo, estatus de fumador y comorbilidades) para predecir el tipo de intervención recomendada: tratamiento endovascular (EVAR) o cirugía abierta.

## Arquitectura General
Este proyecto sigue una arquitectura de monorepositorio dividida claramente en dos entornos principales, garantizando así la separación entre la interfaz de usuario y la lógica algorítmica:

* **Frontend (`/frontend`)**: Una aplicación web interactiva construida con React y Vite. Proporciona a los médicos una interfaz rápida y moderna para introducir datos de los pacientes y visualizar los diagnósticos y gráficas de explicabilidad generadas por los modelos.
* **Backend (`/backend`)**: El núcleo analítico y servidor de la aplicación. Contiene la API de predicción, la canalización de procesamiento de datos (EDA) y los scripts de entrenamiento de los modelos de Machine Learning (en R y Python).

## Objetivos Alcanzados
1.  **Predicción de Riesgo**: Implementación de modelos capaces de evaluar los factores de riesgo a partir de datos clínicos reales.
2.  **Interactividad Web**: Creación de una UI fluida donde el modelo devuelve un diagnóstico instantáneo.
3.  **Explicabilidad**: Conexión directa entre el backend y el frontend para renderizar métricas interpretables de los modelos que justifican el diagnóstico.