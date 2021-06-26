import numpy
import layer


class InputLayer(layer.Layer):

    def __init__(self, p_number_neurons=1):
        layer.Layer.__init__(self, p_number_neurons, 1)

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        # self.w == biases + weight
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=0.5,
                                      size=(1 + self.number_neurons, self.number_neurons))

        self.w = numpy.concatenate((numpy.zeros((1, self.number_neurons)), numpy.eye(self.number_neurons)))

        return self

    def activation(self, activation):
        self.activation = activation

    def predict(self, p_X, index):
        return self._net_input(p_X)
