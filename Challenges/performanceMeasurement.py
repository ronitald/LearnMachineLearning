import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Lectura de los parámetros
filas_columnas, tamanio_entrenamiento = input().split()
filas_columnas = int(filas_columnas)
tamanio_entrenamiento = float(tamanio_entrenamiento)

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Crear el conjunto de datos
data = {
    'Area': np.random.randint(800, 3000, size=filas_columnas),
    'NumeroHabitaciones': np.random.randint(2, 6, size=filas_columnas),
    'NumeroBanios': np.random.randint(1, 4, size=filas_columnas),
    'Barrio': np.random.choice(['A', 'B', 'C'], size=filas_columnas),
    'Edad': np.random.randint(1, 50, size=filas_columnas),
    'PrecioVenta': np.random.randint(100000, 500000, size=filas_columnas)
}

# Convertir el diccionario en un DataFrame
df = pd.DataFrame(data)

# Separar características (X) y la variable objetivo (y)
X = df.drop('PrecioVenta', axis=1)
y = df['PrecioVenta']

# Codificar las variables categóricas
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X[['Barrio']]).toarray()
X = X.drop('Barrio', axis=1)
X = np.hstack((X, X_encoded))

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=tamanio_entrenamiento, random_state=42)

# Entrenar el modelo de Regresión de Lasso
lasso_model = Lasso(alpha=1.0, random_state=42)
lasso_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = lasso_model.predict(X_test)

# Calcular el MSE y el R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir los resultados con dos decimales
print(f"{mse:.2f} {r2:.2f}")
