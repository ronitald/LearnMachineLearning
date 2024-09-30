import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar datos sintéticos para el problema de regresión
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones sobre los datos de prueba
y_pred = model.predict(X_test)

# Calcular la pérdida cuadrática media entre las predicciones y las etiquetas verdaderas
mse_loss = mean_squared_error(y_test, y_pred)
print("Pérdida Cuadrática Media:", mse_loss)