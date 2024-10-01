# Importar el clasificador BaggingClassifier
# Pasting no suele ser utilizado en ejemplos reales. 
# El puntaje de exactitud evidencia que es un pésimo modelo
from sklearn.ensemble import BaggingClassifier
#Genera el objeto a raíz de un árbol de decisión (puede ser el que quieran)
bagging = BaggingClassifier(DecisionTreeClassifier(),
                           #cantidad de (en este caso) árboles de decisión
                           n_estimators=100,
                            #La cantidad de datos que tomará de una muestra
                           max_samples=10,
                           #Bagging=True; Pasting=False
                           bootstrap=True)
bagging.fit(x_train,y_train)
y_pred = bagging.predict(x_test)
#Aunque dará un puntaje similar al clasificador de votaciones, esto se debe al set de datos pequeño.
accuracy_score(y_test,y_pred)

#Hacer el ejemplo con pasting pero utilizar ahora SVC
from sklearn.ensemble import BaggingClassifier
pasting = BaggingClassifier(SVC(),
                           n_estimators=50,
                           max_samples=15,
                           bootstrap=False)
pasting.fit(x_train,y_train)
y_pred = pasting.predict(x_test)
accuracy_score(y_test,y_pred)

# Evaluación Out-of-Bag
#Importar BaggingClassifier
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(DecisionTreeClassifier(),
                           n_estimators=100,
                           max_samples=10,
                            #Poner oob_score=True para poder hacer la evaluación
                           bootstrap=True, oob_score=True)
#Ajustar
bagging.fit(x_train,y_train)
#Visualiza el puntaje OOB
bagging.oob_score_