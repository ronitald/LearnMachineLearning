#Hacer un Bagging como los anteriores
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(DecisionTreeClassifier(),
                           n_estimators=100,
                           max_samples=10,
                           bootstrap=True,
                            #Max_features le dice que solo tomará 2 variables
                           max_features=2,
                           oob_score=True)
#Ajustar
bagging.fit(x_train,y_train)

#Visualizar el puntaje de exactitud
bagging.fit(x_train,y_train)
y_pred=bagging.predict(x_test)
accuracy_score(y_test,y_pred)