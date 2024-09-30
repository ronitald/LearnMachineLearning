# 1) Una red vacía

weight = 0.1
alpha = 0.01

def neural_network(input, weight):
    prediction = input * weight
    return prediction

# 2) PREDICT: Hacer una predicción y evaluar el error

number_of_toes = [8.5]
win_or_lose_binary = [1] # (won!!!)

input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]

pred = neural_network(input,weight)
error = (pred - goal_pred) ** 2

# 3) COMPARAR: Calcular el "delta del nodo" y ponerlo en el nodo de salida
delta = pred - goal_pred

# 4) APRENDE: Calcular el "Delta de Peso" y Ponerlo en el Peso

weight_delta = input * delta

# 5) APRENDE: Actualizar el peso

alpha = 0.01 # fijo antes del entrenamiento
weight -= weight_delta * alpha

weight, goal_pred, input = (0.0, 0.8, 0.5)