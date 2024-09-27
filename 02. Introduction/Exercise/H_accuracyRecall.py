# Importar archivo Main
from G_matriz import *
from E_modelClfBinary import *

# Importar precision_score y recall_score
from sklearn.metrics import precision_score, recall_score
p = precision_score(y_train_y,y_train_pred)
r = recall_score(y_train_y,y_train_pred)
p,r

# Cambiar de clasificador
# Importar RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 42)
rfc.fit(x_train,y_train_y)

#Hacer la matriz de confusión otra vez
y_train_pred = cross_val_predict(rfc,x_train,y_train_y,cv=3)
confusion_matrix(y_train_y,y_train_pred)

# Calcular la precisión y recall con el nuevo clasificador
p = precision_score(y_train_y,y_train_pred)
r = recall_score(y_train_y,y_train_pred)
p,r

# Calcular F1
from sklearn.metrics import f1_score
f1_score(y_train_y,y_train_pred)