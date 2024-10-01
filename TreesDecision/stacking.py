#Importa los regresores necesarios StackingRegressor,LinearRegression, RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Generar el árbol de decisión
arbol = DecisionTreeRegressor(max_depth=20)
#Generar la regresión lineal
lineal = LinearRegression()
#Generar el bosque aleatorio
random = RandomForestRegressor()
#Genera los tres modelos
stacking = StackingRegressor(estimators=[("arbol", arbol),
                            ("lineal", lineal),
                          ("random", random)])
#Utiliza Stacking
stacking.fit(x,y)