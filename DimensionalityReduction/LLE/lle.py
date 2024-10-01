# LLE= Locally Linear Embedding
# Es un método de reducción de dimensionalidad no lineal 
# y no depende de proyecciones
# Funciona midiendo como cada instancia de entrenamiento 
# se relaciona linealmente con sus instancias vecinas y 
# busca una representación lineal de pocas dimensiones del set de datos 
# donde estas relaciones entre instancias cercanas o vecinas 
# están bien preservadas.

#Importar LocallyLinearEmbedding
from sklearn.manifold import LocallyLinearEmbedding

#Seleccionar el número de dimensiones, componentes y "vecinos"
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
x_nueva = lle.fit_transform(x)
x_nueva[0:5]