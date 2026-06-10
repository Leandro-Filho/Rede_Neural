# Otimizador Stochastic Gradient Descent, atualiza os parâmetros da rede usando o parêmtro antigo menos a taxa de apredndizado vezes o gradiente. Essa conta normal de aprendizado

class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(
        self,
        model,
        grad_weights: list,
        grad_biases: list
    ) -> None:

        for layer_index in range(model.num_layers):
            model.weights[layer_index] -= (
                self.learning_rate
                * grad_weights[layer_index]
            )

            model.biases[layer_index] -= (
                self.learning_rate
                * grad_biases[layer_index]
            )

            #Essa parte foi bem tranquila por conta das aulas, mas o código não era igual eu pensava kkkkk