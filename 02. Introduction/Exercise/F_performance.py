# Importar archivo Main
from A_getData import *
from E_modelClfBinary import *

# Realiza una Cross validation/K-Folds
from sklearn.model_selection import cross_val_score
cross_val_score(sgd,x_train,y_train_y,cv=3,scoring="accuracy")

# Modelo que nunca es Y
from sklearn.base import BaseEstimator
class NuncaC(BaseEstimator):
    def fit(self,X,y=None):
        return self
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

nunca_c = NuncaC()
cross_val_score(nunca_c,x_train,y_train_y,cv=3,scoring="accuracy")