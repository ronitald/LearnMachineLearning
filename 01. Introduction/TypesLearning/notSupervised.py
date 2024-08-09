# Importamos las librerías.

# 'matplotlib.pyplot' se utiliza para crear gráficos y visualizaciones en Python.
import matplotlib.pyplot as plt

# 'KMeans' es un algoritmo de clustering que agrupa los datos en 'k' clusters, donde 'k' es un número predefinido.
from sklearn.cluster import KMeans

# 'PCA' (Análisis de Componentes Principales) es una técnica de reducción de dimensionalida.
# Transforma los datos en un espacio de menor dimensión.
from sklearn.decomposition import PCA
from sklearn import datasets

iris = datasets.load_iris()

# Reducimos la dimensionalidad de los datos usando PCA para visualización.
# Queremos reducir las características del conjunto de datos de 4 dimensiones a 2 dimensiones para poder visualizarlo en un gráfico.
pca = PCA(n_components=2)  # 'n_components=2' indica que queremos reducir a 2 dimensiones.
X_reducido = pca.fit_transform(iris.data)  # 'fit_transform' ajusta PCA a los datos y los transforma.

# Inicializamos el modelo K-Means con 3 clusters.
# Aquí especificamos que queremos dividir los datos en 3 grupos (ya que sabemos que hay 3 especies de iris).
kmeans = KMeans(n_clusters=3, random_state=42)  
kmeans.fit(X_reducido)  # Ajustamos el modelo a los datos reducidos.

# Obtenemos las etiquetas de los clusters.
# Las etiquetas indican a qué cluster pertenece cada punto de datos después de aplicar K-Means.
etiquetas = kmeans.labels_

# Visualizamos los resultados en un gráfico de dispersión (scatter plot).
# Los puntos en el gráfico representan las flores de iris y se colorean según el cluster al que pertenecen.
plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=etiquetas, cmap='viridis')
plt.title('Agrupamiento con K-Means')  # Título del gráfico.
plt.show()  # Mostramos el gráfico.
