import numpy as np

from mlp.activations import relu, relu_derivative, softmax # importando as funçõeszinhas de ativação que criamos

# fazendo a classe da rede neural, aqui é onde a mágica acontece, é onde a gente cria a estrutura da rede, os pesos, os bias e o forward pass
class MLP:

    def __init__(self, layer_sizes: list[int], seed: int = 42):

        self.layer_sizes = layer_sizes # a gente passa uma lista com o número de neurônios em cada camada, por exemplo [784, 128, 64, 10] para uma rede com 784 neurônios na camada de entrada, 128 na primeira camada oculta, 64 na segunda camada oculta e 10 na camada de saída (para classificação de dígitos de 0 a 9)
        self.num_layers = len(layer_sizes) - 1 # o número de camadas é o número de conexões entre as camadas, ou seja, se temos 4 camadas (entrada, oculta 1, oculta 2 e saída), temos 3 camadas de pesos (entre entrada e oculta 1, entre oculta 1 e oculta 2, e entre oculta 2 e saída)

        self.weights = [] # pesos do modelo, cada camada tem uma matriz de pesos que conecta os neurônios da camada anterior com os neurônios da camada atual
        self.biases = [] # seria o b do que a gente aprendeu em aula

        self.activations = []
        self.pre_activations = []

        rng = np.random.default_rng(seed) #  aqui criamos números aleatórios para que tenha uma evolução. Se começarmos com 0 em todos os pesos, a rede não ia aprender

        for layer_index in range(self.num_layers): # esse loop vai criar os pesos e os bias para cada camada da rede
            input_size = layer_sizes[layer_index]
            output_size = layer_sizes[layer_index + 1]

            weight_scale = np.sqrt(2.0 / input_size) # define uma escala adequada para redes que usam Relu

            weight_matrix = rng.normal(
                loc=0.0,
                scale=weight_scale,
                size=(input_size, output_size)
            )

            bias_vector = np.zeros((1, output_size)) # inicializa os bias com zeros, acho justo começar assim para não afetar o treinamento e o ajusto dos pesos

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
 # executa o forward pass
    def forward(self, x: np.ndarray) -> np.ndarray:

        self.activations = [x] # guarda a entrada como a primeira ativação
        self.pre_activations = [] # aqui a gente vai guardar os valores antes de aplicar a função de ativação

        current_activation = x

        for layer_index in range(self.num_layers - 1):
            z = (
                current_activation @ self.weights[layer_index] # multiplicação de matrizes
                + self.biases[layer_index] # soma o b
            )

            current_activation = relu(z) # aplica a Relu

            self.pre_activations.append(z)
            self.activations.append(current_activation)

        logits = (
            current_activation @ self.weights[-1] # multiplicação de matrizes para a última camada
            + self.biases[-1] # soma o b da última camada
        )

        probabilities = softmax(logits) # aplica a softmax para obter as probabilidades de cada classe

        self.pre_activations.append(logits) # guarda os logits antes da softmax, pode ser útil para o backpropagation
        self.activations.append(probabilities)

        return probabilities

    # Um desafio da parte de cima foi entender como a estruturação da rede e porque não começar com os pesos em 0. Aprendi depois que fui dar uma olhada do porque todos os pesos tavam iguais depois do forward pass e percebi que era porque os pesos estavam todos iguais, então a rede não tinha como aprender, porque a atualização dos pesos ia ser a mesma para todos os pesos, então a rede não ia conseguir diferenciar os neurônios e aprender coisas diferentes em cada um. Por isso é importante começar com pesos aleatórios, para que a rede possa aprender coisas diferentes em cada neurônio e evoluir de forma mais eficiente

# Faz o backpropagation
    def backward(self, y_true: np.ndarray) -> tuple[list, list]:
 
        num_samples = y_true.shape[0] 

        probabilities = self.activations[-1].copy() # pega as probabilidades da última camada, que é a saída da rede

        one_hot_targets = np.zeros_like(probabilities)
        one_hot_targets[np.arange(num_samples), y_true] = 1

        delta = (probabilities - one_hot_targets) / num_samples # compara o que a rede previu menos o que deveria ter previsto

        grad_weights = [None] * self.num_layers # gradientes das matrizes de pesos
        grad_biases = [None] * self.num_layers # gradientes dos vetores de b

        for layer_index in reversed(range(self.num_layers)): # esse loop vai calcular os gradientes para cada camada, começando da última camada e indo para a primeira
            previous_activation = self.activations[layer_index]

            grad_weights[layer_index] = previous_activation.T @ delta # multiplicação de matrizes para calcular o gradiente dos pesos

            grad_biases[layer_index] = np.sum( # soma os deltas para calcular o gradiente dos b
                delta,
                axis=0,
                keepdims=True
            )

            if layer_index > 0: # se não for a primeira camada, calcula o delta para a próxima camada
                delta = (
                    delta @ self.weights[layer_index].T
                ) * relu_derivative(
                    self.pre_activations[layer_index - 1]
                )

        return grad_weights, grad_biases