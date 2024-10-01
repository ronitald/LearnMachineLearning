# Importamos la libreria
import pandas as pd

# Obtenemos los datos
cantidades = list(map(int, input().split()))
precios = list(map(int, input().split()))
producto_seleccionado = int(input()) - 1  # Convertir a índice (0-4)

# Crear el DataFrame de ventas
datos_ventas = {
    'Fecha': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
    'Producto': ['Laptop', 'Teléfono', 'Tablet', 'Laptop', 'Teléfono'],
    'Cantidad': cantidades,
    'Precio Unitario': precios,
}

df_ventas = pd.DataFrame(datos_ventas)

# Calcular el total de ventas por mes del producto seleccionado
ventas_producto_seleccionado = df_ventas.loc[producto_seleccionado, 'Cantidad'] * df_ventas.loc[producto_seleccionado, 'Precio Unitario']

# Calcular el total de ventas del último año
df_ventas['Total'] = df_ventas['Cantidad'] * df_ventas['Precio Unitario']
total_ventas_ano = df_ventas['Total'].sum()

# Calcular la cantidad total vendida de cada producto
cantidades_totales = df_ventas.groupby('Producto')['Cantidad'].sum()
producto_mas_vendido = cantidades_totales.max()  # Obtener la cantidad máxima vendida

# Calcular el promedio de precios unitarios
promedio_precio_unitario = df_ventas['Precio Unitario'].mean()

# Ventas por mes del producto seleccionado
ventas_por_mes_producto = ventas_producto_seleccionado

# Imprimir los resultados
resultados = [
    ventas_por_mes_producto,
    total_ventas_ano,
    producto_mas_vendido,
    round(promedio_precio_unitario, 2),
    ventas_por_mes_producto
]

# Imprimir los resultados como una cadena de texto
print(" ".join(map(str, resultados)))
