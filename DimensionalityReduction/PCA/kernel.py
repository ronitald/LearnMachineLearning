#Importar KernelPCA
from sklearn.decomposition import KernelPCA
#Hacer un rbf= Radial Basis function
pca_rbf = KernelPCA(n_components=2, kernel="rbf")
x_nueva=pca_rbf.fit_transform(x)
#Visualiza los resultado
x_nueva[0:5]

#Importar GridSearchCV
from sklearn.model_selection import GridSearchCV
#Importar LogisticRegressin
from sklearn.linear_model import LogisticRegression
#Importar Pipeline
from sklearn.pipeline import Pipeline

#Armar un pipeline que pase por KernelPCA y LogisticRegression
clf = Pipeline([
    ("pca", KernelPCA(n_components=2)),
    ("reg", LogisticRegression())
])
#Corre el GridSearch
grid = [{
    "pca__gamma": np.linspace(0.1,0.5,10),
    "pca__kernel": ["rbf","sigmoid"]
}]
#Ejecuta el GridSearch
search = GridSearchCV(clf, grid)
search.fit(x,y)

#Imprime el mejor parámetro que equivale al PCA que será la mejor regresión logística
print(search.best_params_)