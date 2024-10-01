# Importar los bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

url = "https://raw.githubusercontent.com/jofsanchezci/Datos_Int_IA/main/Comediantes.csv"
df=pd.read_csv(url)
# Go es el veredicto de si será o no será comediante
# print(df)

# Utiliza el label_encoder para asignar un valor numérico a Nacionalidad y a Go
from sklearn.preprocessing import LabelEncoder

def label_encoder(datos_categoria):
    le = LabelEncoder()
    df[datos_categoria] = le.fit_transform(df[datos_categoria])

variables = ["Nationality","Go"]

for l in variables:
    label_encoder(l)
# verifica que las variables "Nacionality" y "Go" se cambiaron a valores numéricos

# print(df)

#Separa en dos partes el dataframe
y = df["Go"]
x = df.drop("Go",axis=1)

# verifica que se haya hecho la separación
print(x)
print(y)

#Haz el clasificador de DecisionTree
    #Importa la función
from sklearn.tree import DecisionTreeClassifier
    #Haz el objeto
arbol = DecisionTreeClassifier()
    #Haz el ajuste
arbol.fit(x,y)
DecisionTreeClassifier()

# Gráfica el árbol de decisión
from sklearn import tree
tree.plot_tree(arbol,feature_names=x.columns,rounded=True,filled=True)
# plt.show()

# Probabilidades
# Selecciona una persona aleatoria para estimar su probabilidad y juega con las variables
arbol.predict_proba([[40,6,7,1]])

# Regresión
# Genera 100 número aleatorios y añade un poco de aleatoriedad
m = 100
x = 3 * np.random.rand(m,1) - 3
y = 2 + x + 0.5*x**2 + np.random.rand(m,1)
#np.c_[([x])]
plt.scatter(x,y)

# Crea un  árbol de decisión con un hiperparametro de 2
from sklearn.tree import DecisionTreeRegressor
arbol = DecisionTreeRegressor(max_depth=2)
arbol.fit(x,y)
DecisionTreeRegressor(max_depth=2)

# Dale formato a tu arbol
tree.plot_tree(arbol,rounded=True,filled=True)

# Hacer gráfica
x1 = np.linspace(-3,-2.568)
y1 = np.linspace(3.5818,3.5818)
x2 = np.linspace(-2.569,-2.008)
y2 = np.linspace(2.834,2.834)
x3 = np.linspace(-2.009,-0.151)
y3 = np.linspace(2.075,2.075)
x4 = np.linspace(-0.152,0)
y4 = np.linspace(2.392,2.392)
plt.scatter(x,y)
plt.plot(x1,y1,"r-",linewidth=5)
plt.plot(x2,y2,"r-",linewidth=5)
plt.plot(x3,y3,"r-",linewidth=5)
plt.plot(x4,y4,"r-",linewidth=5)

# plt.show()

#Arreglar el sobreajuste poniendo una profundidad máxima mayor a la anterior. Ej: 3.
from sklearn.tree import DecisionTreeRegressor
arbol = DecisionTreeRegressor(max_depth=3)
arbol.fit(x,y)
#Gráficalo
tree.plot_tree(arbol,rounded=True,filled=True)
#tree.plot_tree(arbol,rounded=True,filled=True,fontsize=15)Para hacer más grande la letra ajustar fontsize

#Haz un árbol de decisiones sin restricciones
from sklearn.tree import DecisionTreeRegressor
arbol = DecisionTreeRegressor()
arbol.fit(x,y)
DecisionTreeRegressor()

#Gráficalo y asómbrate
tree.plot_tree(arbol,rounded=True,filled=True)
plt.show()