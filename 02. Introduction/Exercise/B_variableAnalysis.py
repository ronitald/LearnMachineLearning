# Importar archivo Main
from A_getData import *

# Analiza la edad, visualiza sus máximos y mínimos
print("Max Age:", df.Age.max())
print("Min Age:", df.Age.min())

# Gráfica la variable
# Establece una área de figsize(9,5) es decir, el tamaño de la imagen
plt.figure(figsize = (9,5))

# Crea un displot para de la edad (por ser una variable numérica)
sns.displot(df.Age,kde=True)

# Género: cuenta cuántos hombres y mujeres hay
df.Sex.value_counts()

# Crea una gráfica de barras para Presión Sanguinea
plt.figure(figsize = (9,5))
sns.histplot(data=df,x="BP",hue="BP")

# Crea una gráfica de barras para Colesterol
plt.figure(figsize = (9,5))
sns.histplot(data=df,x="Cholesterol",hue="Cholesterol")

# Crea un displot para Sodio Potasio
plt.figure(figsize = (9,5))
sns.displot(df.Na_to_K,kde=True)

# Crea una gráfica de barras para los Medicamentos (droga) 💊
plt.figure(figsize = (9,5))
sns.histplot(data=df,x="Drug",hue="Drug")

# Cuenta los medicamentos
df.Drug.value_counts()
