# Importar archivo Main
from I_umbral import *

# Importar roc_curve
from sklearn.metrics import roc_curve
fpr, tpr, umbrales = roc_curve(y_train_y,y_scores)

# Graficar la curva ROC
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1],[0, 1], 'k--')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

# Poner una cuadrícula
plt.grid()
plt.show()

# Calcular el puntaje de la curva
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_y,y_scores)

# Compararlo con el modelo de random forest
y_forest = cross_val_predict(rfc,x_train,y_train_y,cv=3,method="predict_proba")
y_scores_forest = y_forest[:,1]

# Calcular el puntaje (área bajo la curva) de random forest
roc_auc_score(y_train_y,y_scores_forest)