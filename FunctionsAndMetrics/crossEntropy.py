import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Calcula la pérdida de entropía cruzada entre las predicciones y las etiquetas verdaderas.

    Args:
    y_true : numpy array
        Etiquetas verdaderas (one-hot encoding)
    y_pred : numpy array
        Predicciones del modelo (probabilidades)

    Returns:
    float
        Pérdida de entropía cruzada
    """
    epsilon = 1e-15  # para evitar divisiones por cero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # evitar valores extremos
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Ejemplo de uso:
y_true = np.array([0, 1, 0])  # Etiqueta verdadera (ejemplo de clasificación binaria)
y_pred = np.array([0.3, 0.6, 0.1])  # Predicción del modelo
ce_loss = cross_entropy_loss(y_true, y_pred)
print("Pérdida de Entropía Cruzada:", ce_loss)