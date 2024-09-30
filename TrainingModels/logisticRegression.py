#Traer el set de datos
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
#Visualizar el set de datos
df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
df.head()

#Dividir el set de datos en en las variables predichas y la variable a predecir
x = df[['work_experience','gpa']]
y = df['admitted']

#importar LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x,y)

# LogisticRegression()

#Hacer las predicciones
y_pred = clf.predict(x)
#Visualiza y_pred
y_pred

#Utilizar la función predict_proba para visualizar la probabilidad de que sea admitido
y_proba = clf.predict_proba(x)
#Generar una lista para ver la probabilidad de ser admitidos y a los que no
y_probano = [y_proba[i][1] for i in range(0,len(y_proba))]
y_probasi = [y_proba[i][0] for i in range(0,len(y_proba))]
#Incluir estas listas en nuestro dataframe
df["predict"] = y_pred
df["probano"] = y_probano
df["probasi"] = y_probasi
df.head()

#Visualizar el peso de la experiencia laboral, de gmat y de gpa
df.sort_values(by=["work_experience"])
df.sort_values(by=["gmat"])
df.sort_values(by=["gpa"])

#Evaluar la regresión con métricas como la matriz de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y)

#Evaluar la regresión con métricas como f1_score
from sklearn.metrics import f1_score
f1_score(y_pred, y)