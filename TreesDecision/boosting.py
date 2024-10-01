#Importar AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
#Crear objeto de Adaboost con árboles de decisiones
ada = AdaBoostClassifier(DecisionTreeClassifier(),
                         n_estimators=100,
                         learning_rate=0.1)
ada.fit(x_train,y_train)

#Calcular el puntaje de exactitud para adaboost
y_pred=ada.predict(x_test)
accuracy_score(y_test,y_pred)
0.9166666666666666

# Boosting con Gradiente
# Crear un set de datos, especificamente una parábola para una mejor visualización
m = 100
x = np.linspace(-0.5,0.5,m)
y = 25*x**2 + np.random.random(m) -0.5
plt.scatter(x,y)

#Crear el primer predictor, utiliza árboles de decisiones y regularizalo con un max_depth=2
from sklearn.tree import DecisionTreeRegressor
x = x.reshape(-1,1)
arbol1 = DecisionTreeRegressor(max_depth=2)
arbol1.fit(x,y)

#Calcular los errores residuales que
#son la diferencia entre las predicciones del modelo y el valor de la variable a predecir
y2 = y - arbol1.predict(x)
#crear el segundo predictor en base a esos residuales.
arbol2 = DecisionTreeRegressor(max_depth=2)
arbol2.fit(x,y2)

#Visualizar que sucede si sumamos las predicciones del primer predictor con el segundo
y_pred = sum(arbol.predict(x) for arbol in (arbol1, arbol2))
#Repetir el mismo procedimiento para crear el árbol 3.
y3 = y - arbol2.predict(x)
arbol3 = DecisionTreeRegressor(max_depth=2)
arbol3.fit(x,y3)

#Ahora vamos a ver que sucede si sumamos los 3 árboles
y_pred = sum(arbol.predict(x) for arbol in (arbol1, arbol2, arbol3))
#Crear exactamente el mismo modelo que creamos antes
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2,
                                n_estimators=3,
                                learning_rate=1.0)
gbrt.fit(x,y)

#En este caso utilizaremos el error medio cuadrado.
from sklearn.metrics import mean_squared_error
x_train, x_test, y_train, y_test = train_test_split(x,y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(x_train, y_train)

errores = [mean_squared_error(y_test,y_pred) for y_pred in gbrt.staged_predict(x_test)]
mejor = np.argmin(errores)
mejor

#Evaluar el early stopping automáticamente con XGBRegressor
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
xgb = XGBRegressor()
xgb.fit(x_train, y_train,
       eval_set=[(x_test,y_test)], early_stopping_rounds=1)
y_pred = xgb.predict(x_test)