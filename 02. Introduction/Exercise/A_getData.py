# Importa las librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Define una función para extraer los datos
# DOWNLOAD_ROOT es la base del GitHub donde vamos a estar descargando las bases de datos.
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/jofsanchezci/Datos_Int_IA/main/drug200.csv"
# Complementos con la dirección especifica de la base de datos que queremos.

def extraer_datos(database):
    csv_path = database
    return pd.read_csv(csv_path)

# Visualiza el DataFrame
df = extraer_datos(DOWNLOAD_ROOT)
df.head()

# Obten información de los datos.
df.info()