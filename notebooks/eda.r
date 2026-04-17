# Análisis exploratorio de los datos para buscar una precisión mayor en el modelo
library(readr)
dataset_clean <- read_delim("universidad/Proyecto-1/data/processed/dataset_clean.csv", 
                            delim = ";", escape_double = FALSE, trim_ws = TRUE)
View(dataset_clean)

# Una vez cargado el dataset, empezaremos a explorar este dataset limpio

# Usaré librerías vistas en la asignatura Modelos de Regresión
# install.packages("repmod")
library(repmod)      # Tablas de resultados de modelos, métricas de rendimiento y validación interna
library(performance) # Validación de modelos estadísticos

modelo_simple = glm(TTO ~ ., data=dataset_clean, family="binomial")
report(modelo_simple)

# Antes de sacar variables, vamos a ver si se cumplen asunciones:
residuos_pearson <- residuals(modelo_simple, type="pearson")
plot(modelo_simple, which=1)
plot(modelo_simple, which=5)

# Limpieza de columnas
# install.packages("effectsize")
library(effectsize)
cohens_f_squared(modelo_simple)


modelo_nuevo = glm(TTO ~ Edad + Fumador + FA, data=dataset_clean, family="binomial")
report(modelo_nuevo)
cohens_f_squared(modelo_nuevo)


nuevo_df = dataset_clean[, c("TTO", "Edad", "Fumador", "FA", "CANCER", "ICC", "EPOC")]
modelo_full <- glm(TTO ~ (Edad + Fumador + FA + CANCER + ICC + EPOC)^2, data = nuevo_df, family = "binomial")
modelo_optimo <- step(modelo_full, direction = "both")

modelo_optimo = glm(TTO ~ Edad + Fumador + FA + CANCER + ICC + EPOC + CANCER:ICC + CANCER:EPOC, data = nuevo_df)




write.csv2(nuevo_df, "universidad/Proyecto-1/data/processed/dataset_clean.csv", row.names = FALSE)
