# Importar archivo Main
from J_curvaROC import *

# Importar SVC=Support Vector Classifier
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)

# Predecir a un humano aleatorio para ver que todo este funcionando bien
svm.predict([[25,0,1,0,167,1]])

# Utilizar decision_function para observar los puntajes de cada medicina
svm.decision_function([[25,0,1,0,167,1,0]])
# Decidirá por el qué tenga mayor puntaje

# Ahora, utiliza el clasificador multiclase
from sklearn.multiclass import OneVsRestClassifier
svm = OneVsRestClassifier(SVC())
svm.fit(x_train,y_train)

# Predecir a un humano ahora con este clasificador
svm.predict([[25,0,1,0,167,1]])

# Utilizar decision_function para observar los puntajes de cada medicina
svm.decision_function([[25,0,1,0,167,1,0]])

# Compararlo con los datos obtenidos de sgd.fit
sgd.fit(x_train,y_train)