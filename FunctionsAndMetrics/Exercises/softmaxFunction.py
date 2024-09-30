# Ejercicio 1: Uso de la Función Softmax

# Enunciado:
# Implementa una red neuronal multicapa utilizando una biblioteca de aprendizaje profundo como TensorFlow o PyTorch. 
# Utiliza la función Softmax como capa de salida y entrena el modelo en el conjunto de datos de imágenes proporcionado. 
# Evalúa el rendimiento del modelo utilizando métricas como la precisión (accuracy) en un conjunto de datos de prueba separado.

# Importar las bibliotecas necesarias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesar los datos
# Normalizar las imágenes de 0-255 a 0-1
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Convertir las etiquetas a formato one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Crear el modelo de red neuronal
model = models.Sequential()
# Primera capa oculta con 128 neuronas y activación ReLU
model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
# Segunda capa oculta con 64 neuronas y activación ReLU
model.add(layers.Dense(64, activation='relu'))
# Capa de salida con función Softmax (10 clases para MNIST)
model.add(layers.Dense(10, activation='softmax'))

# Compilar el modelo con función de pérdida de entropía cruzada y optimizador Adam
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Evaluar el rendimiento en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc:.4f}')
