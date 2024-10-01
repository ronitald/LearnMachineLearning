#Importar IncrementalPCA
from sklearn.decomposition import IncrementalPCA
#Divide los datos en 3, es decir, 3 subsets.
subsets = 3
ipca = IncrementalPCA(n_components=1)
for subset in np.array_split(x, subsets):
    ipca.partial_fit(subset)

x_nueva = ipca.transform(x)
x_nueva[0:5]