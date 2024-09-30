#Importar ElasticNet
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

net = ElasticNet(alpha=0.1, l1_ratio=0.5)
net.fit(x,y)

#Calcular los parámetros de intersección y coeficientes de x
net.intercept_, net.coef_

#Hacer la linea de la predicción de Red Elástica
x_net = np.linspace(0,1,100)
y_net = 4.14013 + 1.5356 * x_net

#graficar comparando la regresión lineal, de Ridge, de Lasso y de Red Elástica
plt.scatter(x,y,label="Datos")
plt.plot(x_nuevo,y_nuevo,"r-",label="Regresión Lineal")
plt.plot(x_cresta,y_cresta,"g-",label="Regresión Ridge")
plt.plot(x_lasso,y_lasso,"b-",label="Regresión Lasso")
plt.plot(x_net,y_net,"y-",label="Regresión Red Elástica")
plt.legend()
plt.show()