# Análisis exploratorio de los datos para buscar una precisión mayor en el modelo
library(readr)
dataset_clean <- read_delim("universidad/Proyecto-1/data/processed/pre_r_df.csv", 
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

# Limpieza de variables
library(glmnet)      # Ajuste de modelos lineales generalizados con penalización (Lasso, Ridge y Elastic Net)
library(mlbench)     # Conjuntos de datos (benchmarks)
library(corrplot)    # Visualización gráfica de matrices de correlación

X <- as.matrix(dataset_clean[,1:15])
y = dataset_clean$TTO

# Volveré a crear el modelo con glmnet
modelo_simple <- cv.glmnet(X, y, family="binomial")
modelo_simple$lambda.1se

modelo_ajustado <- glmnet(X, y, family="binomial")
report(modelo_ajustado, s=modelo_simple$lambda.1se, file="modelo_ajustado", type = "csv")

predicciones <- predict(modelo_ajustado, newx = X, s=modelo_simple$lambda.1se)
AUC(predicciones, y)

write.csv2(modelo_ajustado, "universidad/Proyecto-1/data/processed/dataset_clean.csv", row.names = FALSE)

