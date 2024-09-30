import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calcula la pérdida cuadrática media entre las etiquetas verdaderas y las predicciones.

    Args:
    y_true : numpy array
        Etiquetas verdaderas
    y_pred : numpy array
        Predicciones del modelo

    Returns:
    float
        Pérdida cuadrática media
    """
    return np.mean((y_true - y_pred) ** 2)

# Ejemplo de uso:
y_true = np.array([3, -0.5, 2, 7])  # Etiquetas verdaderas
y_pred = np.array([2.5, 0.0, 2, 8])  # Predicciones del modelo
mse_loss = mean_squared_error(y_true, y_pred)
print("Pérdida Cuadrática Media:", mse_loss)