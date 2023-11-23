import numpy as np


class Perceptron:

    def __init__(self, lr, n_epochs):
        # Construtor, define taxa de aprendizado e número máximo de épocas
        self.w_bias = 0
        self.total_error = []
        self.bias = 0
        self.weights = []
        self.lr = lr
        self.n_epochs = n_epochs

    @staticmethod
    def activation(value):
        return 1 if value >= 0 else -1

    def predict(self, x):
        return np.dot(x, self.weights.T) + self.bias * self.w_bias

    @staticmethod
    def evaluate(target, predicted):
        return target - predicted

    def train(self, x, d):

        self.weights = np.random.random(x.shape[1])
        self.bias = np.random.random()
        self.w_bias = np.random.random()

        epoch = 0
        is_error = True
        self.total_error = []

        while is_error and epoch < self.n_epochs:

            is_error = False
            epoch_errors = 0

            # Para cada amostra
            for xi, target in zip(x, d):

                predicted = self.predict(xi)
                predicted = self.activation(predicted)

                current_error = self.evaluate(target, predicted)
                epoch_errors += current_error

                # Se houve erro, atualizar os pesos
                if predicted != target:
                    self.weights += self.lr * current_error * xi
                    self.w_bias += self.lr * current_error * self.bias
                    is_error = True

            self.total_error.append(epoch_errors / len(x))
            epoch += 1

    def test(self, x):
        results = []
        for xi in x:
            predict = self.predict(xi)
            predict = self.activation(predict)
            results.append(predict)

        return results
