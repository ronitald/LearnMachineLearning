# PCA=Principal Component Analysis

#Importa los librerías pandas, numpy, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#Trae los datos que necesitamos
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
df.head()

#Separa el dataframe en los datos que vamos a utilizar para predecir y los datos predichos
x = df[['gpa','gmat','work_experience']]
y = df['admitted']
#Una vez teniendo nuestros datos centramos x restándole su media.
x_centrada  = x - x.mean(axis=0)
#Recordemos que SVD descompone X en 3 matrices U, E y V por lo que utilizamos la función de numpy svd.
U, E, V = np.linalg.svd(x_centrada)
#V es la que contiene los vectores con los componentes principales
#para obtener los primeros 2 simplemente transponemos sus primeras 2 columnas.
pc1 = V.T[:,0]
pc2 = V.T[:,1]
pc1,pc2

W = V.T[:, :2]
x_nueva = x_centrada.dot(W)
x_nueva.head()

