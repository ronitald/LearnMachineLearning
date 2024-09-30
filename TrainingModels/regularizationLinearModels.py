import numpy as np
import matplotlib.pyplot as plt

#Hacer un set de juguete
x = np.random.rand(100,1)
y = 3 + 3 * x + np.random.rand(100,1)
plt.scatter(x,y,label="Datos")

#Hacer una regresión lineal sencilla para comparar con la regresión de cresta
lin_reg=LinearRegression()
lin_reg.fit(x,y)
lin_reg.intercept_, lin_reg.coef_

#Generar 100 datos para graficar la linea de predicción
x_nuevo = np.linspace(0,1,100)
y_nuevo = 3.46 + 3.09 * x_nuevo

#graficar
plt.scatter(x,y,label="Datos")
plt.plot(x_nuevo,y_nuevo,"r-",label="Regresión Lineal")
plt.legend()
plt.show()

#Importar Ridge para hacer nuestra regresión de cresta
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(x,y)

#Calcula los parámetros de ridge
ridge.intercept_, ridge.coef_

#Hacer la linea de la predicción de Ridge
x_cresta = np.linspace(0,1,100)
y_cresta = 3.66 + 2.72 * x_cresta

#graficar comparando la regresión lineal y de Ridge
plt.scatter(x,y,label="Datos")
plt.plot(x_nuevo,y_nuevo,"r-",label="Regresión Lineal")
plt.plot(x_cresta,y_cresta,"g-",label="Regresión Ridge")
plt.legend()
plt.show()