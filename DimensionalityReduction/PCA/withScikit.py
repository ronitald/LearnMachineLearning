#Importa PCA
from sklearn.decomposition import PCA
#Genera el objeto
pca = PCA(n_components=2)
x_nueva = pca.fit_transform(x)
x_nueva[0:5]

#Calula la distribución de la varianza
pca.explained_variance_ratio_

# Crea  un modelo donde tengas al menos el 90% de la varianza consevada
pca = PCA(n_components=0.90)
x_nueva = pca.fit_transform(x)
x_nueva[0:5]

#descomprime los datos con la función inverse_transform
pca = PCA(n_components=1)
x_nueva = pca.fit_transform(x)
x_recuperada = pca.inverse_transform(x_nueva)
#Resultado de la descompresión
x_recuperada[0:5]

