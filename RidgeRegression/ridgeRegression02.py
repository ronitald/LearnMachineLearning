#Hacer una regresión lineal sencilla para comparar con la regresión de cresta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
poli = PolynomialFeatures(degree=2, include_bias=False)
x_poli = poli.fit_transform(x)
x[0], x_poli[0]



regresion_polinomial = Pipeline([
    ("carac_polinomiales", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg_lin", LinearRegression())
])
lin_reg=LinearRegression()
lin_reg.fit(x,y)
lin_reg.intercept_, lin_reg.coef_

#Importar Ridge para hacer nuestra regresión de cresta
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(x,y)

#Calcula los parámetros de ridge
ridge.intercept_, ridge.coef_

#Hacer la linea de la predicción de Ridge
x_cresta = np.linspace(0,1,100)
y_cresta = 3.66 + 2.72 * x_cresta
     

#Generar 100 datos para graficar la linea de predicción
x_nuevo = np.linspace(0,1,100)
y_nuevo = 3.46 + 3.09 * x_nuevo
     

#graficar comparando la regresión lineal y de Ridge
plt.scatter(x,y,label="Datos")
plt.plot(x_nuevo,y_nuevo,"r-",label="Regresión Lineal")
plt.plot(x_cresta,y_cresta,"g-",label="Regresión Ridge")
plt.legend()
plt.show()

#Importar lasso
from sklearn.linear_model import Lasso
#Asignar una alpha de 0.1
lasso = Lasso(alpha=0.1)
lasso.fit(x,y)

#Calcular los parámetros de intersección y coeficientes de x
lasso.intercept_, lasso.coef_

#graficar comparando la regresión lineal, de Ridge y de Lasso
plt.scatter(x,y,label="Datos")
plt.plot(x_nuevo,y_nuevo,"r-",label="Regresión Lineal")
plt.plot(x_cresta,y_cresta,"g-",label="Regresión Ridge")
plt.plot(x_lasso,y_lasso,"b-",label="Regresión Lasso")
plt.legend()
plt.show()

# Red Elastica

#Importar ElasticNet
from sklearn.linear_model import ElasticNet
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