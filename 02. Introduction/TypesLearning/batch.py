import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento: tamaño de la vivienda (en metros cuadrados) y precio (en miles de dólares)
X = np.array([[50], [60], [70], [80], [90]])  # Tamaño de la vivienda
y = np.array([150, 180, 210, 240, 270])       # Precio de la vivienda

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo usando todos los datos (aprendizaje batch)
model.fit(X, y)

# Realizar una predicción
size_new_house = np.array([[85]])  # Tamaño de una nueva vivienda
predicted_price = model.predict(size_new_house)

print(f"Predicción del precio para una vivienda de 85 m²: ${predicted_price[0]:.2f} mil dólares")
