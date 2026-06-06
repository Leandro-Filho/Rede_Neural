import numpy as np # impotamos os numpy para fazer as contas/operações 

# Aqui ele aplica a Relu, nada mais é que: se o valor foi zeero ou positivo, ele retorna o valor, se for negativo, ele retorna zero.
def relu(x: np.ndarray) -> np.ndarray:

    return np.maximum(0, x) # ele pega o valor máximo entre 0 e x da entrada que entegamos para ele, fazendo a função relu de fato

# Aqui calcula a derivada da função Relu, ou seja, ele retorna 1 para valores positivos e 0 para valores negativos ou iguais a zero.
def relu_derivative(x: np.ndarray) -> np.ndarray:

    return (x > 0).astype(float) # aqui é simples: se x é maior que 0, retorna true e o astype faz a função de transformar em 1 ou 0.

# O desafio aqui, mesmo que o cáculo seja bem simples, foi pensar mesmo em como transformar isso em código.