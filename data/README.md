# Gestión de Datos - Proyecto Aneurismas

Esta carpeta contiene el ciclo de vida de los datos del proyecto.

## Estructura
- **/raw**: Datos originales del hospital (`RawData.csv`).
  - *Nota*: Contiene un 80% de valores nulos en la columna de diámetro.
- **/processed**: Datos listos para el modelo.
  - `processed_data.csv`: Dataset completo (~400 registros). Sin columna de diámetro por falta de datos.
  - `alternative_data.csv`: **Subconjunto Dorado**. Solo pacientes con diámetro registrado (~100 registros).

## Pipeline de Limpieza
El script `data_preparation.py` realiza:
1. Filtrado de columnas con >40% de nulos.
2. Mapeo de la variable objetivo `TTO` (0: No IQ, 1: Abierta, 2: EVAR).
3. Conversión de tipos (Booleanos para patologías y Float para diámetros).