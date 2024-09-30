import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generar datos sintéticos para la regresión
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión de cresta
alpha = 1  # Parámetro de regularización
ridge_reg = Ridge(alpha=alpha)
ridge_reg.fit(X_train, y_train)

# Coeficientes del modelo
print("Coeficientes del modelo:", ridge_reg.coef_)
print("Término de intercepción:", ridge_reg.intercept_)

# Predecir sobre los datos de prueba
y_pred = ridge_reg.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)

# Visualizar los datos y la línea de regresión
plt.scatter(X, y, color='blue', label='Datos')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Regresión de Cresta')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Regresión de Cresta')
plt.show()

x = np.random.rand(100,1)
y = 3 + 3 * x + np.random.rand(100,1)
plt.scatter(x,y,label="Datos")