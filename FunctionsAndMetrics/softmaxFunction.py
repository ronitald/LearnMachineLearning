import numpy as np

def softmax(x):
    """
    Calcula la función Softmax para un vector de entrada x.

    Args:
    x : numpy array
        Vector de entrada

    Returns:
    numpy array
        Vector de probabilidades normalizado
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

# Ejemplo de uso:
x = np.array([1.0, 2.0, 3.0])
softmax_output = softmax(x)
print("Salida de Softmax:", softmax_output)