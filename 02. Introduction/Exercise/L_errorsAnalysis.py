from K_multiclassClassifier import *

# Hacer un clasificador de random forest
rfc.fit(x_train, y_train)

# Utilizar la matriz de confusión
y_train_pred = cross_val_predict(rfc, x_train, y_train, cv=3)
conf_mz = confusion_matrix(y_train,y_train_pred)
conf_mz

# Utilizar ahora SGD
y_train_pred = cross_val_predict(sgd, x_train, y_train, cv=3)
conf_mz = confusion_matrix(y_train,y_train_pred)
conf_mz