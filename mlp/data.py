# esse código sera responsável por carregar os dados do MNIST, preparar os dados para o treinamento e teste da rede neural. Ele também pode conter funções para pré-processamento dos dados, como normalização ou transformação de rótulos.
import numpy as np

from keras.datasets import mnist


def load_mnist_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Carrega e prepara o dataset MNIST.

    Retorna:
        x_train: imagens de treino normalizadas
        y_train: respostas corretas do treino
        x_test: imagens de teste normalizadas
        y_test: respostas corretas do teste
    """
    (x_train, y_train), (x_test, y_test) = (
        mnist.load_data()
    )

    x_train = x_train.reshape(
        x_train.shape[0],
        -1
    )

    x_test = x_test.reshape(
        x_test.shape[0],
        -1
    )

    x_train = x_train.astype(
        np.float32
    ) / 255.0

    x_test = x_test.astype(
        np.float32
    ) / 255.0

    return (
        x_train,
        y_train,
        x_test,
        y_test
    )