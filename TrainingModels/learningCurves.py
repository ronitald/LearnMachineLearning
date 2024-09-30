#Importar mean_squared_error train_test_split para medir el error sobre los datos de entranamiento y validación
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def curvas_aprendizaje(modelo ,x , y):
    #Empezamos dividiendo los datos en datos de entrenamiento y validación
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    #Generar una lista vacías para irlas rellenando conforme se vaya calculando los errores
    train_error, val_errors= [], []
    #Tomar el set de entrenamiento y ajustándolo al modelo pero solo con un dato de entrenamiento y así sucesivamente
    for m in range(1,len(x_train)):
        modelo.fit(x_train[:m], y_train[:m])
        #predecir el modelo
        y_train_pred = modelo.predict(x_train[:m])
        #predecir el modelo
        y_val_predict = modelo.predict(x_val)
        #Calcular los errores
        train_error.append(mean_squared_error(y_train[:m],y_train_pred))
        val_errors.append(mean_squared_error(y_val,y_val_predict))

        #graficarlos
    plt.plot(np.sqrt(train_error), "r-+", label = "set de entranamiento")
    plt.plot(np.sqrt(val_errors), "b-", label = "set de validación")
    plt.axis([0,80,0,2])
    plt.legend()

#Correr la curva de aprendizaje
reg_lin = LinearRegression()
curvas_aprendizaje(reg_lin, x, y)

#Hacer un pipeline llamado regresion_polinomial que haga una regresión polinomial y lineal
from sklearn.pipeline import Pipeline

regresion_polinomial = Pipeline([
    ("carac_polinomiales", PolynomialFeatures(degree=10, include_bias=False)),
    ("reg_lin", LinearRegression())
])
#Ejecutar curvas de aprendizaje a regresion_polinomial
curvas_aprendizaje(regresion_polinomial,x,y)

#Variar el grado del polinomio para mejorar el rendimiento del modelo. Ejemplo:2
from sklearn.pipeline import Pipeline

regresion_polinomial = Pipeline([
    ("carac_polinomiales", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg_lin", LinearRegression())
])
curvas_aprendizaje(regresion_polinomial,x,y)

