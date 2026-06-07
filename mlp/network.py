import numpy as np

from mlp.activations import relu, softmax # importando as funçõeszinhas de ativação que criamos

# fazendo a classe da rede neural, aqui é onde a mágica acontece, é onde a gente cria a estrutura da rede, os pesos, os bias e o forward pass
class MLP:

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        self.weights = [] # pesos do modelo, cada camada tem uma matriz de pesos que conecta os neurônios da camada anterior com os neurônios da camada atual
        self.biases = [] # seria o b do que a gente aprendeu em aula

        self.activations = []
        self.pre_activations = []

        rng = np.random.default_rng(seed)

        for layer_index in range(self.num_layers):
            input_size = layer_sizes[layer_index]
            output_size = layer_sizes[layer_index + 1]

            weight_scale = np.sqrt(2.0 / input_size)

            weight_matrix = rng.normal(
                loc=0.0,
                scale=weight_scale,
                size=(input_size, output_size)
            )

            bias_vector = np.zeros((1, output_size))

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Executa o forward pass.

        Recebe os dados de entrada e devolve
        as probabilidades previstas pela rede.
        """
        self.activations = [x]
        self.pre_activations = []

        current_activation = x

        for layer_index in range(self.num_layers - 1):
            z = (
                current_activation @ self.weights[layer_index]
                + self.biases[layer_index]
            )

            current_activation = relu(z)

            self.pre_activations.append(z)
            self.activations.append(current_activation)

        logits = (
            current_activation @ self.weights[-1]
            + self.biases[-1]
        )

        probabilities = softmax(logits)

        self.pre_activations.append(logits)
        self.activations.append(probabilities)

        return probabilities