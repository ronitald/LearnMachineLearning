# Ejercicio 3: Uso de la Pérdida Cuadrática Media

# Enunciado:
# Implementa un modelo de regresión lineal utilizando una biblioteca de aprendizaje automático como Scikit-Learn. 
# Utiliza la pérdida cuadrática media como función de costo y entrena el modelo en el conjunto de datos de ventas proporcionado. 
# Evalúa el rendimiento del modelo utilizando métricas como el error cuadrático medio en un conjunto de datos de prueba separado.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Datos de ventas (puedes comentar esta sección si prefieres cargar desde un archivo)
data = {
    'publicidad': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'ventas': [150, 250, 350, 450, 550, 650, 700, 800, 850, 1000]
}

# Crear un DataFrame a partir de los datos
df = pd.DataFrame(data)

# Separar las características (X) y la variable objetivo (y)
X = df[['publicidad']]  # Característica: gasto en publicidad
y = df['ventas']        # Variable objetivo: ventas

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular la pérdida cuadrática media
mse = mean_squared_error(y_test, y_pred)

print(f'Error Cuadrático Medio: {mse:.2f}')

# Si prefieres cargar los datos desde un archivo, puedes usar el siguiente código:
# data = pd.read_csv('ventas.csv')
