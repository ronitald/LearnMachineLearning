# 1) Una red vacía

weight = 0.1
lr = 0.01

def neural_network(input, weight):
    prediction = input * weight
    return prediction


# 2) PREDICT: Hacer una predicción y evaluar el error

number_of_toes = [8.5]
win_or_lose_binary = [1] #(won!!!)

input = number_of_toes[0]
true = win_or_lose_binary[0]

pred = neural_network(input,weight)
error = (pred - true) ** 2
print(error)

# 3) COMPARE: Hacer una predicción con una ponderación *mayor* y evaluar el error

weight = 0.1

def neural_network(input, weight):
    prediction = input * weight
    return prediction

number_of_toes = [8.5]
win_or_lose_binary = [1] #(won!!!)

input = number_of_toes[0]
true = win_or_lose_binary[0]

lr = 0.01
p_up = neural_network(input,weight+lr)
e_up = (p_up - true) ** 2
print(e_up)

# 4) COMPARE: Hacer una predicción con una ponderación *más baja* y evaluar el error

weight = 0.1

def neural_network(input, weight):
    prediction = input * weight
    return prediction

number_of_toes = [8.5]
win_or_lose_binary = [1] #(won!!!)

input = number_of_toes[0]
true = win_or_lose_binary[0]

lr = 0.01
p_dn = neural_network(input,weight-lr)
e_dn = (p_dn - true) ** 2
print(e_dn)

