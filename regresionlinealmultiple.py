# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:14:59 2022

@author: ariel
"""

#Regresion Lineal Multiple 

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values 


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
X = np.array(ct.fit_transform(X), dtype=np.float)

#Evitar la trampa de las variables ficticias

X = X[:, 1:] #sacamos la primera columna

#Dividir el data set en conjunto de entrenamiento y testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 0)

#Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de los resultados en el conjunto de testing

y_pred = regression.predict(X_test)

#Construir el modelo optimo de RLM utilizando la eliminacion hacia atras
import statsmodels.api as sm
#import statsmodels.formula.api as sm
#Agregamos una columna de 1's para representar el coef de posicion en la recta
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#Creamos la matriz de caracteristicas optimas para nuestro modelo
#Metodo eliminacion hacia atras

#Iteracion 1
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(y, X_opt).fit()


#Calculamos los p para cada variable para decidir cual eliminar

regressor_OLS.summary()

#Iteracion 2

X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(y, X_opt).fit()


#Calculamos los p para cada variable para decidir cual eliminar

regressor_OLS.summary()

#Iteracion 3 

X_opt = X[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(y, X_opt).fit()


#Calculamos los p para cada variable para decidir cual eliminar

regressor_OLS.summary()

#Iteracion 4

X_opt = X[:, [0, 3, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(y, X_opt).fit()


#Calculamos los p para cada variable para decidir cual eliminar

regressor_OLS.summary()

#Iteracion 5

X_opt = X[:, [0, 3]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(y, X_opt).fit()


#Calculamos los p para cada variable para decidir cual eliminar

regressor_OLS.summary()

#Se transformo en un modelo de regresion lineal simple

#Automatizacion
#####################################################
def backwardElimination(X, sl):    
    numVars = len(X[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, X_opt).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    X = np.delete(X, j, 1)    
    regressor_OLS.summary()    
    return X 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)






