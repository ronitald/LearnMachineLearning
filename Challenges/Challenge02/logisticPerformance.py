# Medición del Rendimiento de un Modelo de Clasificación con Regresión Logística

# Importar bibliotecas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos de cáncer de mama
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data 
y = data.target # (0: No cáncer, 1: Cáncer)

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de regresión logística
clf = LogisticRegression(max_iter=10000, random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular y mostrar la precisión del modelo
accuracy = np.mean(y_pred == y_test)
print(f'Precisión: {accuracy:.2f}')

# Predicciones sobre el conjunto de prueba
y_pred = clf.predict(X_test)

# Informe de clasificación
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:\n', conf_matrix)

# Visualización de la matriz de confusión
plt.matshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.ylabel('Valor verdadero')
plt.xlabel('Valor predicho')
plt.show()

# Validación cruzada para una evaluación más robusta del modelo
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f'Precisión media de validación cruzada: {cv_scores.mean():.2f}')

# Generar la curva ROC
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f'Área bajo la curva (AUC): {roc_auc:.2f}')

# Visualización de la curva ROC
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Línea de referencia
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='best')
plt.show()
