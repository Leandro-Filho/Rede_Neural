import numpy as np

# Essa função vai calcular o quanto foi o erro da rede, ou seja, o quanto a rede errou em relação a resposta correta
def cross_entropy_loss(probabilities: np.ndarray, y_true: np.ndarray) -> float:

    # Matriz com as probabilidades previstas pela rede.
    num_samples = probabilities.shape[0]

    # essa parte é bem interessante: para não ter valores igual a zerp dentro do log, colocamos que o valor mínimo é 1e-12, isso ainda possibilita a gente fazer as contas 
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)

    correct_class_probabilities = clipped_probabilities[
        np.arange(num_samples),
        y_true #  Vetor com a classe correta de cada amostra.
    ]

    # log do erro 
    losses = -np.log(correct_class_probabilities)

    return np.mean(losses)

    # um desafio nessa parte foi entender como resolver o problema do log 0, entender a lógica de o vator verdadeiro e o que a rede coisou