#Importar RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
#Hacer RandomForestClassifier, establecer los parámetros similares al bagging
rf = RandomForestClassifier(n_estimators=100,
                           max_leaf_nodes=4,
                           max_features=2)
#Hacer el ajuste
rf.fit(x_train, y_train)

#Hacer las predicciones
y_pred=rf.predict(x_test)

#Visualiza el puntaje
accuracy_score(y_test,y_pred)

# Predictores
#Realiza el árbol de manera aleatoria
random = RandomForestClassifier(n_estimators=100)
#Despliega los parámetros importantes o predictores
random.fit(x_train,y_train)
for nombre, score in zip(x.columns, random.feature_importances_):
    print(nombre,score)