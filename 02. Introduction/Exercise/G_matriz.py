# Importar archivo Main
from E_modelClfBinary import *

# Matriz de confusión
# Importar cross_val_predict
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd,x_train,y_train_y,cv=3)

#Importar confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_y,y_train_pred)