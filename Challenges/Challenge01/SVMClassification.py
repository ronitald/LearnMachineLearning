# Clasificación de Datos con Máquinas de Vectores de Soporte (SVM)
# Importar las bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()

# Filtrar solo dos clases para la clasificación
X = iris.data[iris.target != 2, :2] 
y = iris.target[iris.target != 2]

# Dividir el conjunto en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo SVM con kernel lineal
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular y mostrar la precisión del modelo
accuracy = np.mean(y_pred == y_test)
print(f'Precisión: {accuracy:.2f}')

# Informe de clasificación
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:\n', conf_matrix)

# Graficar los datos y la línea de separación
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)
plt.xlabel('Largo del Sépalo')
plt.ylabel('Ancho del Sépalo')

# Obtener coeficientes y punto de intersección para la línea de separación
coef = clf.coef_
intercept = clf.intercept_

# Calcular y dibujar la línea de separación
xlim = plt.xlim()  # Límites en el eje x
xx = np.linspace(xlim[0], xlim[1])  # Puntos en el eje x
yy = -(coef[0, 0] * xx + intercept[0]) / coef[0, 1]  # Ecuación de la línea

plt.plot(xx, yy, 'k--')  # Línea de separación
plt.title('Clasificación SVM en el conjunto de datos Iris (solo dos clases)')
plt.show()  # Mostrar la gráfica
