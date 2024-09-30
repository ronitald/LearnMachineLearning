import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# epochs: cuántas gradientes hará, cuántos datos seleccionará para hacer el gradiente
epochs = 50
#Calendario de aprendizaje, sirve para definir el ritmo de aprendizaje
t0 , t1 = 5, 50
m = 100
#Definir el horario de aprendizaje
def horario_aprendizaje(t):
    return t0 / (t + t1)
#Definir los 2 valores con lo que van a empezar
b = np.random.rand(2,1)

#Definir la función
for epoch in range(epochs):
    for i in range(m):
        #Genera un índice aleatorio
        random_index = np.random.randint(m)
        #Valor x de la coordenada que seleccionamos al azar
        xi = x_b[random_index:random_index+1]
        #Valor y de la coordenada que seleccionamos al azar
        yi = y[random_index:random_index+1]
        #Calcular el gradiente (Resultado de la derivada parcial)
        gradientes  = 2 * xi.T.dot(xi.dot(b) - yi)
        #Calcular el ritmo de aprendizaje
        ra = horario_aprendizaje(epoch * m + i)
        #Calcular los parámetros de intersección y pendiente
        b = b -ra * gradientes
b

# #Repetir el ejercicio pero desplegando cada gradiente realizado
epochs = 4
t0 , t1 = 5, 50
m = 5

def horario_aprendizaje(t):
    return t0 / (t + t1)

b = np.random.rand(2,1)
#Agrega un scatterplot para ver los datos
plt.scatter(x,y,label="Datos")

for epoch in range(epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        y_nuevo = x_nuevo_b.dot(b)
        #Grafica las líneas rojas que simbolizan los diferentes gradientes a través de las iteraciones.
        plt.plot(x_nuevo,y_nuevo,"r-")
        gradientes  = 2 * xi.T.dot(xi.dot(b) - yi)
        ra = horario_aprendizaje(epoch * m + i)
        b = b -ra * gradientes
plt.show()

#Hacerlo con scikit
from sklearn.linear_model import SGDRegressor
#La toleración es el límite menor al valor de la suma de los errores al cuadrado
sgd = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
#.ravel: te genera una lista normal
sgd.fit(x, y.ravel())
#Sacar los parámetros, la intersección y la pendiente
sgd.intercept_, sgd.coef_

#Visualiza la lista normal que genera .ravel
y.ravel()