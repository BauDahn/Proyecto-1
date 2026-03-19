# Código Fuente y Lógica de IA

Implementación del modelo de predicción de tratamiento para aneurismas.

## Archivos Principales
- **`tree.py`**: Implementación desde cero de un Árbol de Decisión usando Impureza de Gini.
- **`data_preparation.py`**: Script de preprocesamiento de datos.
- **`validation.py`**: Suite de pruebas con **Bootstrap (200 iteraciones)**.

## Uso Técnico
La función `evaluar_con_bootstrap` ha sido parametrizada para permitir pruebas rápidas:
```python
# Ejemplo de uso
media, std = evaluar_con_bootstrap(X, y, max_depth=5, min_samples=10)