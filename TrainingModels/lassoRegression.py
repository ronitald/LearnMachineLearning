#Importar lasso
from sklearn.linear_model import Lasso
#Asignar una alpha de 0.1
lasso = Lasso(alpha=0.1)
lasso.fit(x,y)

#Calcular los parámetros de intersección y coeficientes de x
lasso.intercept_, lasso.coef_

#Hacer la linea de la predicción de lasso
x_lasso = np.linspace(0,1,100)
y_lasso = 3.99 + 1.86 * x_lasso

#graficar comparando la regresión lineal, de Ridge y de Lasso
plt.scatter(x,y,label="Datos")
plt.plot(x_nuevo,y_nuevo,"r-",label="Regresión Lineal")
plt.plot(x_cresta,y_cresta,"g-",label="Regresión Ridge")
plt.plot(x_lasso,y_lasso,"b-",label="Regresión Lasso")
plt.legend()
plt.show()