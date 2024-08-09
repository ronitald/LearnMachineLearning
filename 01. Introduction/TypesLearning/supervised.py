# Importar las librerías necesarias para trabajar con Machine Learning
# 'sklearn' es una biblioteca de Python utilizada para realizar tareas de machine learning.

# Esta función se utiliza para dividir un conjunto de datos en dos subconjuntos: entrenamiento y prueba.
from sklearn.model_selection import train_test_split 

# Este clasificador es un método de aprendizaje supervisado.
from sklearn.neighbors import KNeighborsClassifier 

# Este módulo proporciona varios conjuntos de datos predefinidos.
from sklearn import datasets 

# Cargamos el conjunto de datos 'iris'.
# El conjunto de datos 'iris' es un dataset muy conocido que contiene información sobre 3 tipos de flores de iris.
# Este dataset incluye 4 características para cada flor, como la longitud y el ancho de los sépalos y pétalos.
iris = datasets.load_iris() 

# Separamos el conjunto de datos en dos componentes:
# X: las variables de entrada (también conocidas como características o features). 
# y: la variable objetivo (también conocida como etiqueta o label) que queremos predecir.
X = iris.data  # Aquí almacenamos las características de las flores (longitud del sépalo, ancho del sépalo, etc.)
y = iris.target  # Aquí almacenamos la clase de cada flor (0, 1, 2, donde cada número representa una especie de flor).

# Dividimos los datos en dos subconjuntos: entrenamiento y prueba.
# train para entrenar el modelo y test para probar la precisión del modelo.
# El argumento 'test_size=0.2' indica que el 20% de los datos se utilizarán para pruebas y el 80% para entrenamiento.
# El argumento 'random_state=42' se utiliza para garantizar que la división sea reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Creamos una instancia del clasificador K-Nearest Neighbors.
# Aquí configuramos el clasificador para que utilice k=3, es decir, que considere los 3 vecinos más cercanos para hacer una predicción.
clf = KNeighborsClassifier(n_neighbors=3) 

# Entrenamos (ajustamos) el modelo utilizando los datos de entrenamiento.
# El método 'fit' ajusta el modelo a los datos de entrada (X_train) y las etiquetas correspondientes (y_train).
clf.fit(X_train, y_train) 

# Evaluamos la precisión del modelo utilizando el conjunto de datos de prueba.
# El método 'score' mide qué tan bien predice el modelo en comparación con las etiquetas reales del conjunto de prueba.
accuracy = clf.score(X_test, y_test)

# Imprimimos la precisión del modelo. 
# La precisión es la proporción de predicciones correctas hechas por el modelo.
print(f'Precisión del modelo: {accuracy}')
