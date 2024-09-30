import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir la tasa de aprendizaje (ra)
ra = 0.1
#Definir las iteraciones. 1000 es un estándar. En 1000 se va a detener
iteraciones = 1000
#Número de datos
m = 100

#Incializa la pendiente
b = np.random.rand(2,1)

#Hacer el programa para el descenso del gradiente
for iteracion in range(iteraciones):
    #Expresión a manera de ágebra lineal de los mínimos cuadrados (función de costo)
    gradientes  = 2/m * x_b.T.dot(x_b.dot(b) - y)
    b = b -ra * gradientes
b

#Repetir el ejercicio pero con ritmo de tasa de aprendizaje diferente
ra = 0.1
iteraciones = 10
m = 100

b = np.random.rand(2,1)

plt.scatter(x,y,label="Datos")

#Visualiza los modelos que va proponiendo hasta llegar a la predicción final
for iteracion in range(iteraciones):
    gradientes  = 2/m * x_b.T.dot(x_b.dot(b) - y)
    b = b -ra * gradientes
    y_nuevo = x_nuevo_b.dot(b)
    plt.plot(x_nuevo,y_nuevo,"r-")
plt.show()
gradientes