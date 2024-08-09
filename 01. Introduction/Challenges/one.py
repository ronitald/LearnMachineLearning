from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Datos de ejemplo (edades y sus categorías)
edades = [5, 10, 12, 13, 15, 17, 18, 22, 25, 30]
categorias = ['Niño', 'Niño', 'Niño', 'Adolescente', 'Adolescente', 
              'Adolescente', 'Adulto', 'Adulto', 'Adulto', 'Adulto']

# Convertimos las categorías a números
mapa_categorias = {'Niño': 0, 'Adolescente': 1, 'Adulto': 2}
categorias_numericas = [mapa_categorias[categoria] for categoria in categorias]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split([[edad] for edad in edades], categorias_numericas, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {precision*100:.2f}%')

# Clasificar nuevas edades
nuevas_edades = [4, 17, 18, 30, 21]
clasificaciones = model.predict([[edad] for edad in nuevas_edades])

# Convertir las clasificaciones a nombres de categorías
clasificaciones_nombre = [list(mapa_categorias.keys())[list(mapa_categorias.values()).index(clas)] for clas in clasificaciones]

# Imprimir las clasificaciones
for edad, clasificacion in zip(nuevas_edades, clasificaciones_nombre):
    print(f'Edad: {edad} - Clasificación: {clasificacion}')
