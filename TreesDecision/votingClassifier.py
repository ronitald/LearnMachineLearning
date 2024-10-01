#Importa las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Importa los clasificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#Este VotingClassifier va a recibir los modelos y va a contar los votos de los resultados
from sklearn.ensemble import VotingClassifier

#Trae el set de datos
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
#Es sencillo pero, es trabajo honesto
df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
df.head()

#Dividir el dataframe en dos: los datos a predecir y los datos predichos
x = df[['gpa','gmat','work_experience']]
y = df['admitted']

#Importa el train_test_split
from sklearn.model_selection import train_test_split
#Divide los datos
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42, shuffle = True)
#Entrena los modelos
    #Genera los objetos
svm = SVC()
    #Establece una profundidad máxima de 2 en el árbol de decisión
arbol = DecisionTreeClassifier(max_depth=2)
#Genera un objeto para el votingClassifier
#En caso de querer una votación suave, poner en voting="Soft"
votos = VotingClassifier(estimators=[ ("svm", svm), ("arbol", arbol)],voting="hard")
votos.fit(x_train,y_train)

#Calcular la exactitud de los modelos con accuracy_score
from sklearn.metrics import accuracy_score
#Ciclar a tráves de los 3 modelos que estamos desarrolando
for i in (svm, arbol, votos):
    #Ajustar
    i.fit(x_train,y_train)
    #Predecir los datos del x_test
    y_pred = i.predict(x_test)
    print(i.__class__.__name__,
        #Arrojar el puntaje de exactitud
         accuracy_score(y_test,y_pred))
    
#En caso de querer un margen suave se tiene que hacer ciertas modificaciones
svm = SVC(probability=True)
votos = VotingClassifier(estimators=[ ("svm", svm), ("arbol", arbol)],voting="soft")
votos.fit(x_train,y_train)
#Calcular la exactitud de los modelos con accuracy_score
from sklearn.metrics import accuracy_score
#Ciclar a tráves de los 3 modelos que estamos desarrolando
for i in (svm, arbol, votos):
    #Ajustar
    i.fit(x_train,y_train)
    #Predecir los datos del x_test
    y_pred = i.predict(x_test)
    print(i.__class__.__name__,
        #Arrojar el puntaje de exactitud
         accuracy_score(y_test,y_pred))