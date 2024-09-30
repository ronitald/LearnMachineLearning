# Ejercicio 2: Uso de la Entropía Cruzada

# Enunciado: Implementa un modelo de clasificación binaria utilizando una biblioteca de aprendizaje profundo como TensorFlow o PyTorch. 
# Utiliza la entropía cruzada como función de costo y entrena el modelo en el conjunto de datos de reseñas de películas proporcionado. 
# Evalúa el rendimiento del modelo utilizando métricas como precisión, recall y F1-score en un conjunto de datos de prueba separado.

# Importar las bibliotecas necesarias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import precision_score, recall_score, f1_score

# Cargar el conjunto de datos IMDB
# Solo tomamos las 10,000 palabras más frecuentes
max_features = 10000
maxlen = 500  # Limitar las reseñas a 500 palabras
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

# Preprocesar los datos: rellenar las secuencias para que tengan la misma longitud
train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)

# Crear el modelo de red neuronal para clasificación binaria
model = models.Sequential()
# Capa de embedding para representar las palabras
model.add(layers.Embedding(max_features, 128, input_length=maxlen))
# Capa de LSTM para capturar dependencias secuenciales
model.add(layers.LSTM(64))
# Capa densa con activación Sigmoid para la clasificación binaria
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo utilizando la entropía cruzada binaria como función de pérdida
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluar el rendimiento en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc:.4f}')

# Hacer predicciones en el conjunto de prueba
predictions = (model.predict(test_data) > 0.5).astype('int32')

# Calcular precisión, recall y F1-score
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print(f'Precisión: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
