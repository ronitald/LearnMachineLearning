import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Generar set de datos de juguete. Añade un toque de aleatoriedad 🧙🏻‍♀️
m = 100
x = 3 * np.random.rand(m,1) - 3
#Elevar la ecuación al 2
y = 2 + x + 0.5*x**2 + np.random.rand(m,1)
#Gráfica el set de datos
plt.scatter(x,y)

#Importar PolynomialFeatures con una potencia 2
from sklearn.preprocessing import PolynomialFeatures
poli = PolynomialFeatures(degree=2, include_bias=False)
x_poli = poli.fit_transform(x)
x[0], x_poli[0]

#Hacer una regresión lineal sobre de los datos
reg_lin = LinearRegression()
reg_lin.fit(x_poli,y)
reg_lin.intercept_,reg_lin.coef_

#Graficar la predicción de PolynomialFeatures y los datos originales
x_nuevo = np.linspace(-3.2,0.2,100)
#escribir la fórmula a partir del array de arriba
y_nuevo = 2.56 + 1.11 * x_nuevo + 0.539 * x_nuevo ** 2
plt.plot(x_nuevo,y_nuevo,"r-",label="Predicción")
plt.scatter(x,y,label="Datos")
plt.legend()
plt.show()