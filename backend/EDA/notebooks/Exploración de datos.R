# Exploración de los datos
library(readr)
dataset_clean <- read_delim("../data/processed/pre_r_df.csv", 
View(dataset_clean)

# Summary del dataset
str(dataset_clean)

# Conversión de tipos
dataset_clean$TTO <- as.logical(dataset_clean$TTO)
dataset_clean$Hospital <- as.factor(dataset_clean$Hospital)
dataset_clean$Fumador <- as.factor(dataset_clean$Fumador)

str(dataset_clean)

summary(dataset_clean)


library(ggplot2)

# Comprobación de variables booleanas

# Sexo
ggplot(dataset_clean, aes(x = TTO, fill = Sexo)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# Dislipemia
ggplot(dataset_clean, aes(x = TTO, fill = Dislipemia)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# Diabetes
ggplot(dataset_clean, aes(x = TTO, fill = Diabetes)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# HTA
ggplot(dataset_clean, aes(x = TTO, fill = HTA)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# IAM
ggplot(dataset_clean, aes(x = TTO, fill = IAM)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# ERC
ggplot(dataset_clean, aes(x = TTO, fill = ERC)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# EPOC
ggplot(dataset_clean, aes(x = TTO, fill = EPOC)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# ACV
ggplot(dataset_clean, aes(x = TTO, fill = ACV)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# FA
ggplot(dataset_clean, aes(x = TTO, fill = FA)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# CANCER
ggplot(dataset_clean, aes(x = TTO, fill = CANCER)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# ICC
ggplot(dataset_clean, aes(x = TTO, fill = ICC)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# EAP
ggplot(dataset_clean, aes(x = TTO, fill = EAP)) + 
  geom_bar(position= "fill") +
  labs(y = "Proporción", x = "TTO", fill= "bool") +
  theme_minimal()

# Comprobación del Hospital (variable Categórica)
dataset_clean$Hospital[dataset_clean$Hospital == 0] <- 1 # Hay un hospital con un 0 que tiene que ser un 1 en el dataset (fila 179)
# Gráfico de barras
ggplot(dataset_clean, aes(x = Hospital, fill = as.factor(TTO))) +
  geom_bar(position = "fill") +
  labs(y = "Proporción", title = "Reparto de TTO por Hospital") +
  theme_minimal()


# Comprobarción de la edad
ggplot(dataset_clean, aes(x = TTO, y = Edad, fill = TTO)) +
  geom_boxplot() +
  labs(title = "Distribución de Edad por Tipo de TTO",
       x = "Tratamiento (TTO)",
       y = "Edad (Años)") +
  theme_minimal()

head(dataset_clean)

# Prueba de suma de comorbilidades como variable categórica
comorbilidades <- c('Dislipemia', 'Diabetes', 'HTA', 'IAM', 'ERC', 'EPOC', 'ACV', 'FA', 'CANCER', 'ICC', 'EAP')

dataset_clean$comorbilidades <- as.factor(rowSums(dataset_clean[, comorbilidades], na.rm = TRUE))

# Comprobación de la utilidad de la variable nueva creada
# Gráfico de barras
ggplot(dataset_clean, aes(x = comorbilidades, fill = as.factor(TTO))) +
  geom_bar(position = "fill") +
  labs(y = "Proporción", title = "Reparto de TTO por Cantidad de Comorbilidades") +
  theme_minimal()

# Voy a agrupar las comorbilidades en grupos
library(forcats)

dataset_clean$comorb_grupos <- fct_collapse(as.factor(dataset_clean$comorbilidades),
                                            "Bajo"  = c("0", "1", "2"),
                                            "Medio" = c("3", "4", "5"),
                                            "Alto"  = c("6", "7", "8")
)
ggplot(dataset_clean, aes(x = comorb_grupos, fill = as.factor(TTO))) +
  geom_bar(position = "fill") +
  labs(y = "Proporción", title = "Reparto de TTO por Cantidad de Comorbilidades") +
  theme_minimal()


columnas_deseadas <- c("Edad", "Sexo", "Fumador", "comorb_grupos", "TTO")
nuevo_dataset <- dataset_clean[, columnas_deseadas]

str(nuevo_dataset)

write.csv2(nuevo_dataset, "backend/EDA/data/processed/dataset_clean.csv", row.names = FALSE)

library(repmod)
library(performance) 


modelo_simple = glm(TTO ~ Edad + Sexo + Fumador + comorb_grupos, data=dataset_clean, family="binomial")
report(modelo_simple)

# Validación del modelo
dataset_clean <- as.data.frame(dataset_clean)

modelo_agrupado <- glm(TTO ~ Edad + Sexo + Fumador + comorb_grupos, 
                       data = dataset_clean, 
                       family = "binomial")

AUCs <- cv_model(TTO ~ Edad + Sexo + Fumador + comorb_grupos, 
                 data = dataset_clean, 
                 fit_function = "glm", 
                 family = binomial())

mean(AUCs)

# Encontrar el mejor umbral
probabilidades <- predict(modelo_agrupado, newdata = test_data, type = "response")

objeto_roc <- roc(test_data$riesgo, probabilidades)
mejor_umbral <- coords(objeto_roc, "best", ret = c("threshold", "sensitivity", "specificity"), best.method = "youden")
print(mejor_umbral)
#Pesos del modelo
pesos <- coef(modelo_agrupado)

tabla_pesos <- data.frame(
  Variable = names(pesos),
  Peso = as.numeric(pesos)
)

write.csv(tabla_pesos, "backend/src/models/model_weights.csv", row.names = FALSE)
