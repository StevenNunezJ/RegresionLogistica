############################################################
# Trabajo en grupo: 
# René Rubio, Jean Mejicanos, Steven Núñez
#
# MBD - Regresión logística
############################################################

# Descripción de las variables en el conjunto de datos
# status: Estado de la cuenta de cheques existente 
# duration: Duración en meses 
# credit.hist: Historial de crédito 
# purpose: Propósito del crédito 
# credit.amo: Monto del crédito 
# savings: Cuenta de ahorros / bonos 
# employed.time: Tiempo de empleo actual 
# installment: Tasa de pago en porcentaje del ingreso 
# status.sex: Estado civil y sexo 
# debtors: Otros deudores / garantes 
# residence.time: Tiempo de residencia 
# property: Propiedad 
# age: Edad 
# installment2: Otros planes de pago 
# housing: Tipo de vivienda 
# num.credits: Número de créditos existentes 
# job: Tipo de empleo 
# num.people: Número de personas a cargo 
# telephone: Teléfono 
# foreign: Trabajador extranjero 
# y: Resultado de pago del crédito
############################################################

############################################################
# Instalación de librerías (solo ejecutar si no están instaladas)
############################################################
install.packages("ggplot2")
install.packages("reshape2")
install.packages("ggcorrplot")
install.packages("caret")
install.packages("ResourceSelection")
install.packages("AUC")
install.packages("PresenceAbsence")

############################################################
# Importación de librerías necesarias
############################################################
library(ggplot2)
library(reshape2)
library(ggcorrplot)
library(caret)
library(ResourceSelection)
library(AUC)
library(PresenceAbsence)

############################################################

# Parte 1. Construir modelo

############################################################


##Limpiar objetos en memoria
rm(list=ls())

# Configurar el directorio de trabajo y cargar los datos
setwd('C:/Users/Steven_Nunez/Documents/Steven/Trabajos Universidad/Universidad LaSalle - Barcelona/Estadistica/Datos y Código-20241023')                                   # directorio de trabajo
datos <- read.table('bank0_train.txt',header=TRUE,sep=';',stringsAsFactors = TRUE) # lectura de los datos


# Exploración de datos inicial
View(datos)               # Ver los datos en una tabla
dim(datos)                # Dimensiones del dataset
head(datos)               # Primeras filas del dataset
summary(datos)            # Estadísticas descriptivas
str(datos)                # Estructura del dataset
sapply(datos, function(x) length(unique(x))) # Valores únicos por columna



# Gráfico de barras para la variable (y)
table(datos$y)
counts <- table(datos$y)
barplot_heights <- barplot(counts, main = "Distribucion del pago del credito", 
                           col = c("lightblue", "salmon"), names.arg = c("No", "Sí"), 
                           ylab = "Frecuencia", ylim = c(0, max(counts) * 1.2))

text(barplot_heights, counts, labels = counts, pos = 3, cex = 1) # Etiquetas encima de cada barra

#Graficos de variables clave
ggplot(datos, aes(x = purpose)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Distribución del propósito del crédito", x = "Propósito", y = "Frecuencia") +
  theme_minimal()

ggplot(datos, aes(x = status)) +
  geom_bar(fill = "lightcoral") +
  labs(title = "Distribución del estado de cuenta", x = "Estado", y = "Frecuencia") +
  theme_minimal()

ggplot(datos, aes(x = status.sex)) +
  geom_bar(fill = "lightgreen") +
  labs(title = "Distribución por estado civil y sexo", x = "Estado Civil y Sexo", y = "Frecuencia") +
  theme_minimal()


# Relación entre edad y monto del crédito
ggplot(datos, aes(x = age, y = credit.amo)) +
  geom_point(color = "blue") +
  labs(title = "Relación entre edad y monto del crédito", x = "Edad", y = "Monto del Crédito") +
  theme_minimal()


#Grafico entre duracion y monton del credito 
ggplot(datos, aes(x = duration, y = credit.amo)) +
  geom_point(color = "purple") +
  labs(title = "Relación entre duración y monto del crédito", x = "Duración en meses", y = "Monto del Crédito") +
  theme_minimal()


# Matriz de correlación para variables numéricas
numeric_data <- datos[sapply(datos, is.numeric)]
cor_matrix <- cor(numeric_data, use = "complete.obs")
ggcorrplot(cor_matrix, lab = TRUE, title = "Matriz de correlación de variables numéricas")



# Descriptiva bivariante
############################################################
##-- Variables categoricas
sapply(datos,class)
var.cat <- which(sapply(datos,class)=="factor" & names(datos)!="y")  # variables que son categoricas (factores)
##--Mosaicplot para las categoricas
for(vc in var.cat)  mosaicplot(datos[,vc]~datos$y,main=names(datos)[vc],col=2:3,las=1)


##--Densidad para las variables numericas
var.num <- which(sapply(datos,class) %in% c("numeric","integer"))    # variables que son numericas
for(vn in var.num) cdplot(datos$y~datos[,vn],main=names(datos)[vn],n=512)


############################################################
# Preparación del modelo
############################################################

##-- La instruccion relevel cambia la categoria de referencia en las variables categoricas
##-- Se escoge la referencia no por criterios estadisticos, sino de interpretabilidad
datos$status <- relevel(datos$status,ref="no checking account")
datos$credit.hist <- relevel(datos$credit.hist,ref="no credits taken/all credits paid back duly")
datos$purpose <- relevel(datos$purpose,ref="others")
datos$savings <- relevel(datos$savings,ref="unknown/no savings account")
datos$employed.time <- relevel(datos$employed.time,ref="unemployed")
datos$status.sex <- relevel(datos$status.sex,ref="male:single")
datos$debtors <- relevel(datos$debtors,ref="none")
datos$property <- relevel(datos$property,ref="unknown/no property")
datos$installment2 <- relevel(datos$installment2,ref="none")
datos$housing <- relevel(datos$housing,ref="for free")
datos$job <- relevel(datos$job,ref="unskilled - resident")
datos$telephone <- relevel(datos$telephone,ref="none")
datos$foreign <- relevel(datos$foreign,ref="no")

# Estimar modelo de regresión logística completo
mod.glm0 <- glm(y~.,datos,family=binomial)    # estimacion del modelo
summary(mod.glm0)                             # resumen del modelo


############################################################
# Seleccion automatica
############################################################
##-- Missings
apply(apply(datos,2,is.na),2,sum)             # cuantos missings tiene cada variable --> Step no se puede aplicar con missings

# Selección de variables (stepwise)
##-- Funcion step (requiere que no haya missings si ha de quitar una variable que tenga)
mod.glm1 <- step(mod.glm0)
summary(mod.glm1)

##-- Importancia de las categorias/covariables segun su significacion estadistica
# Importancia de las variables seleccionadas
View(varImp(mod.glm1))

############################################################
# Validacion del modelo 
############################################################


# Gráfico de pagos observados vs. esperados
br <- quantile(fitted(mod.glm1),seq(0,1,0.1))                                # Se crean los puntos de corte para los intervalos (br) de las probabilidades predichas
int <- cut(fitted(mod.glm1),br)                                              # Se crea una variable con el intervalo al que pertenece cada individuo
obs <- tapply(mod.glm1$y,int,sum)                                            # Los pagos observados en cada intervalo
exp <- tapply(fitted(mod.glm1),int,sum)                                      # Los pagos esperados en cada intervalo  
plot(1:10+0.05,exp,type='h',xlab="Intervalos",ylab="Frecuencias",lwd=2)      # Grafico de los pagos esperados
lines(1:10-0.05,obs,type='h',col=2,lwd=2)                                    # Se anyade los pagos observados
legend("topleft",c("Pagan - esperados", "Pagan - observados"),lwd=2,col=1:2) # Se anyade una leyenda

# test de Hosmer-Lemeshow
hoslem.test(mod.glm1$y, fitted(mod.glm1))  # si el p-valor es inferior a 0.05 quedaria en duda el modelo                              


############################################################
# Estimacion de un Odds Ratio
############################################################
##-- Variable categorica
exp(mod.glm1$coef["foreignyes"])    # Los extranjeros tienen un oddsratio de 0.17 respecto a los no extranjeros. Es decir, las probabilidades de pagar son un 0.17 la de los nacionales 

##-- Variable numerica
exp(mod.glm1$coef["age"])           # Por cada anyo de mas de la persona se incrementa en un 2% (aprox) la probabilidad de que acabe pagando

##--Intervalos de confianza
IC <- confint(mod.glm1)             # Intervalos de  confianza para los coeficientes
round(exp(IC),2)                    # Intervalos de confianza para los ORs redondeados a 2 decimales

############################################################
# Estimacion de la probabilidad de pago
############################################################
##-- Probabilidades predichas
pr <- predict(mod.glm1,datos,type="response")
pr

##--Probabilidad maxima y minima
pos.max <- which.max(pr)        # posicion del individuo con mayor probabilidad de pagar
pr[pos.max]                     # probabilidad de dicho individuo 
datos$y[pos.max]                # pago?

pos.min <- which.min(pr)        # posicion del individuo con menor probabilidad de pagar
pr[pos.min]                     # probabilidad de dicho individuo 
datos$y[pos.min]                # pago?

boxplot(pr~y,datos)

############################################################
# Análisis de Curva ROC y AUC
############################################################

##-- Curva ROC para el conjunto de entrenamiento

# Calcular predicciones y curva ROC para datos de entrenamiento
pr_train <- predict(mod.glm1, type='response')  # Predicciones para el conjunto de entrenamiento
roc_train <- roc(pr_train, datos$y)             # Curva ROC para el conjunto de entrenamiento

# Graficar la curva ROC con el AUC en el título y agregar una línea de referencia
plot(roc_train, col = "blue", lwd = 2, main = paste("Curva ROC - AUC:", round(AUC::auc(roc_train), 2)))
abline(a = 0, b = 1, col = "gray", lty = 2)  # Línea diagonal de referencia (AUC = 0.5)

# Agregar leyenda
legend("bottomright", legend = c("Modelo ROC", "Referencia"), col = c("blue", "gray"), lwd = 2, lty = c(1, 2))

# Análisis de sensibilidad y especificidad para distintos puntos de corte
sensitivity_train <- AUC::sensitivity(pr_train, datos$y)
specificity_train <- AUC::specificity(pr_train, datos$y)
df_train <- data.frame(Cutpoints = sensitivity_train$cutoffs, Sensitivity = sensitivity_train$measure, Specificity = specificity_train$measure)
print(df_train)  # Muestra los puntos de corte, sensibilidad y especificidad en la consola



############################################################
# Calibracion del modelo
############################################################

# Calibración de predicciones
df.calibra <- data.frame(plotID=1:nrow(datos), Observed = as.numeric(datos$y)-1  , Predicted1 = pr)
calibration.plot(df.calibra, N.bins = 10,ylab='Observed probabilities')
detach('package:PresenceAbsence')



############################################################
# Parte 2. Testear resultados
############################################################

#Cargar datos de prueba
test <- read.table('bank0_test.txt',header=TRUE,sep=';', stringsAsFactors = TRUE)

############################################################
# Calcular predicciones y compararlas con valores reales
############################################################
pr <- predict(mod.glm1,test)   # probabilidades predichas
boxplot(pr~test$y)             # Como son estas probabilidades en ambos grupos de respuesta
roc.curve <- roc(pr,test$y)    # Calculo de la curva ROC
plot(roc.curve)                # Dibujo de la curva ROC
AUC::auc(roc.curve)                 # AUC de la curva ROC

# Calcular predicciones y curva ROC para datos de prueba
pr_test <- predict(mod.glm1, test)            # Predicciones para el conjunto de prueba
roc_test <- roc(pr_test, test$y)              # Curva ROC para el conjunto de prueba

# Graficar la curva ROC con el AUC en el título y agregar una línea de referencia
plot(roc_test, col = "red", lwd = 2, main = paste("Curva ROC - AUC:", round(AUC::auc(roc_test), 2)))
abline(a = 0, b = 1, col = "gray", lty = 2)  # Línea diagonal de referencia (AUC = 0.5)

# Agregar leyenda
legend("bottomright", legend = c("Modelo ROC", "Referencia"), col = c("red", "gray"), lwd = 2, lty = c(1, 2))

# Análisis de sensibilidad y especificidad para distintos puntos de corte
sensitivity_test <- AUC::sensitivity(pr_test, test$y)
specificity_test <- AUC::specificity(pr_test, test$y)
df_test <- data.frame(Cutpoints = sensitivity_test$cutoffs, Sensitivity = sensitivity_test$measure, Specificity = specificity_test$measure)
print(df_test)  # Muestra los puntos de corte, sensibilidad y especificidad en la consola


##-- Sensibilidad y especificidad para un punto de corte concreto
s <- AUC::sensitivity(pr,test$y)
e <- AUC::specificity(pr,test$y)
a <- AUC::accuracy(pr,test$y)
df <- data.frame(cutpoints=s$cutoffs,sens=s$measure,esp=e$measure,acc=a$measure)
View(round(df,3))

##-- Escoger un punto de corte --> Matriz de confusion
test$doy.credito <- ifelse(pr>0.5,'si','no')  # Doy credito a aquellos con un probabilidad predicha de pagar superior a 0.5
with(test,table(doy.credito,y))
with(test,round(100*prop.table(table(doy.credito,y),1),1))

 

