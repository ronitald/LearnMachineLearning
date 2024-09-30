# Importar las librerías, numpy, matplotlib, pandas
# numpy es la bibleoteca por exelencia de matemáticas para python
import numpy as np
# Importar matplotlib porque vamos a estar haciendo gráficas
import matplotlib.pyplot as plt
import pandas as pd

# Generar el set de jueguete de datos lineales aleatorios
x = np.random.rand(100,1)

# Genera vector de valores que vas a estar prediciendo (añade un factor de aleatoriedad)
y = 3 + 3 * x + np.random.rand(100,1)

# Gráfica los datos del set de juguete
plt.scatter(x,y)
plt.show()

# Agregar el valor de x0
x_b = np.c_[np.ones((100,1)),x]
# Aplicar la ecuación normal
param = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
# Visualiza
param

# Agregar el valor de x0
x_b = np.c_[np.ones((100,1)),x]
# Aplicar la ecuación normal
param = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
# Visualiza
param

#Grafica la regresión con los datos originales y la predicción
plt.plot(x_nuevo,y_pred,"r-",label="Predicción")
plt.scatter(x,y,label="Datos")
plt.legend()
plt.show()

# Regresor Linear de Scikit
# Haz la regresión con scikit
from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
reg_lin.fit(x,y)
# Cálcula la intersección y la pendiente con este método
reg_lin.intercept_,reg_lin.coef_