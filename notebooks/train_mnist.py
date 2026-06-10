import numpy as np

from mlp.data import load_mnist_data
from mlp.losses import cross_entropy_loss
from mlp.network import MLP
from mlp.optimizers import SGD

# calculamos a acurácia do modelo, ou seja, a parte correta das previsões dividido por todas as previsões.
def calculate_accuracy(
    model: MLP, # puxamos a classe MLP
    x: np.ndarray, # as imagens
    y_true: np.ndarray # as respostas corretas
) -> float:
    probabilities = model.forward(x) # pegamos o Forward, código que fizemos 

# as probabilidades são pegas e tranformadas em previsões, ou seja, o índice da classe com a maior probabilidade é a previsão do modelo
    predictions = np.argmax(
        probabilities,
        axis=1
    )
# comparamos as previsões com as respostas corretas e calculamos a acurácia, que é a média de acertos
    return np.mean(
        predictions == y_true
    )


def train(
    model: MLP,
    optimizer: SGD,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42
) -> None:
    """
    Treina a rede usando SGD com mini-batches.
    """
    rng = np.random.default_rng(seed)

    num_samples = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        indices = np.arange(
            num_samples
        )

        rng.shuffle(indices)

        x_train_shuffled = x_train[
            indices
        ]

        y_train_shuffled = y_train[
            indices
        ]

        epoch_losses = []

        for start_index in range(
            0,
            num_samples,
            batch_size
        ):
            end_index = (
                start_index + batch_size
            )

            x_batch = x_train_shuffled[
                start_index:end_index
            ]

            y_batch = y_train_shuffled[
                start_index:end_index
            ]

            probabilities = model.forward(
                x_batch
            )

            loss = cross_entropy_loss(
                probabilities,
                y_batch
            )

            grad_weights, grad_biases = (
                model.backward(
                    y_batch
                )
            )

            optimizer.step(
                model,
                grad_weights,
                grad_biases
            )

            epoch_losses.append(
                loss
            )

        average_loss = np.mean(
            epoch_losses
        )

        train_accuracy = calculate_accuracy(
            model,
            x_train,
            y_train
        )

        test_accuracy = calculate_accuracy(
            model,
            x_test,
            y_test
        )

        print(
            f"Época {epoch:2d}/{epochs} "
            f"| Loss: {average_loss:.4f} "
            f"| Acurácia treino: {train_accuracy:.2%} "
            f"| Acurácia teste: {test_accuracy:.2%}"
        )


x_train, y_train, x_test, y_test = (
    load_mnist_data()
)

model = MLP(
    layer_sizes=[
        784,
        128,
        64,
        10
    ],
    seed=42
)

optimizer = SGD(
    learning_rate=0.15
)

train(
    model=model,
    optimizer=optimizer,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    epochs=30,
    batch_size=32
)