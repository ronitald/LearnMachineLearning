# Importar las bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# Cargar los datos de detección de fraudes en transacciones financieras
# URL: https://www.kaggle.com/datasets/ealaxi/paysim1
url = ""
df = pd.read_csv(url)

# Procesamiento de los datos
# Limpieza de los datos
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Codificar variables categóricas
df = pd.get_dummies(df, drop_first=True)

# Verificar si todas las columnas son numéricas
print(df.info())

# Dividir los datos en características (X) y etiqueta (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear el modelo de Bosques Aleatorios
clf_rf = RandomForestClassifier(random_state=42)

# Entrenar el modelo
clf_rf.fit(X_train, y_train)

# Evaluar el modelo
y_pred_rf = clf_rf.predict(X_test)

# Métricas de rendimiento para Bosques Aleatorios
print(f'Reporte de clasificación para Bosques Aleatorios:\n {classification_report(y_test, y_pred_rf)}')
print(f'Matriz de confusión para Bosques Aleatorios:\n {confusion_matrix(y_test, y_pred_rf)}')

# Comparación con Regresión Logística
clf_lr = LogisticRegression(max_iter=10000)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
print('Reporte de clasificación para Regresión Logística:\n', classification_report(y_test, y_pred_lr))

# Comparación con SVM
clf_svm = SVC(probability=True)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print('Reporte de clasificación para SVM:\n', classification_report(y_test, y_pred_svm))

# Validación cruzada para Bosques Aleatorios
cv_scores_rf = cross_val_score(clf_rf, X, y, cv=5)
print(f'Precisión media de validación cruzada para Bosques Aleatorios: {cv_scores_rf.mean():.2f}')

# Ajuste de hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(clf_rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print('Mejores hiperparámetros:', grid_search.best_params_)

# Curva ROC para Bosques Aleatorios
y_pred_proba_rf = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label=f'Bosques Aleatorios (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='best')
plt.show()
