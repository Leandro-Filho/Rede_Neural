import numpy as np # impotamos os numpy para fazer as contas/operações 

# Aqui ele aplica a Relu, nada mais é que: se o valor foi zeero ou positivo, ele retorna o valor, se for negativo, ele retorna zero.
def relu(x: np.ndarray) -> np.ndarray:

    return np.maximum(0, x) # ele pega o valor máximo entre 0 e x da entrada que entegamos para ele, fazendo a função relu de fato

# Aqui calcula a derivada da função Relu, ou seja, ele retorna 1 para valores positivos e 0 para valores negativos ou iguais a zero.
def relu_derivative(x: np.ndarray) -> np.ndarray:

    return (x > 0).astype(float) # aqui é simples: se x é maior que 0, retorna true e o astype faz a função de transformar em 1 ou 0.

# O desafio aqui, mesmo que o cáculo seja bem simples, foi pensar mesmo em como transformar isso em código.

# Essa função faz a questão de probabilidade, então ela soma os valores de cada linha para dar a probabilidade
def softmax(x: np.ndarray) -> np.ndarray:

    shifted_x = x - np.max(x, axis=1, keepdims=True) # subtraí o maior value para não termos problemas na exponencial, isso é uma técnica comum para evitar overflow numérico.

    exp_x = np.exp(shifted_x) # aplicamos a exponecial para cada elemento da matriz para transformar em probabilidade

    return exp_x / np.sum(exp_x, axis=1, keepdims=True) # retorna a oorcentagem tirada da divisão entre a exponecial e a soma de todas as exponeciais, isso é a função softmax de fato.

# um problema que tive foi não ter tratado os valores mais altos, então quando eu vou carcular uma exponecial de um valor altissímo, o numpy só retorna "inf", assim um número infinito mais outro número infinito ele retorna nan
#para resolver isso, eu subtraí o maior valor da matriz, assim a exponecial de um número grande se torna a exponecial de um número pequeno, evitando o problema do overflow numérico.