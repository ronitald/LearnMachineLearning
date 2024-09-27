# Importar archivo Main
from D_dataCleaning import *

# Crear modelo para medicamento
y_train_y = (y_train == 0)
y_test_y = (y_test == 0)

# Modelo SGD = Stochastic Gradient Descent (próximamente)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
sgd.fit(x_train,y_train_y)
sgd.predict([[47,1,1,0,8,0]])

# Predecir la medicina a tomar de un humano que ya sepas el resultado
sgd.predict([x_train.loc[0]]), y_train_y.loc[42]