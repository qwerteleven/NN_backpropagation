import input_layer
import hidden_layer
import output_layer
import numpy


class Network(object):
    """Class Network:

       Attributes:
         eta.- Learning rate
         number_iterations.-
         ramdon_state.- Random process seed
         input_layer_.-
         hidden_layers_.-
         output_layer_.-
         sse_while_fit_.-

    """

    def __init__(self, p_eta=0.01, epoch=50, p_random_state=None):
        self.eta = p_eta
        self.epoch = epoch
        self.random_seed = numpy.random.RandomState(p_random_state)


    def init_weigth(self, activation):
        # inicializacion random de los pesos de la red
        for layer, acti in zip(self.network_, activation):
            layer.init_w(self.random_seed)
            layer.activation(acti)

                                                              # vector with size of all hidden layers
    def network_construct(self, input_shape, output_shape, hidden_layers, neurons_hidden_layers, activation):
        self.input_layer_ = input_layer.InputLayer(input_shape)
        self.hidden_layers_ = []

        for v_layer in range(hidden_layers):
            if v_layer == 0:
                self.hidden_layers_.append(hidden_layer.HiddenLayer(neurons_hidden_layers[v_layer],       # output size
                                                                    self.input_layer_.number_neurons))    # input  size
            else:
                self.hidden_layers_.append(hidden_layer.HiddenLayer(neurons_hidden_layers[v_layer],       # output size
                                                                    neurons_hidden_layers[v_layer - 1]))  # input  size

        self.output_layer_ = output_layer.OutputLayer(output_shape,                                       # output size
                                                     self.hidden_layers_[-1].number_neurons)              # input  size

        self.network_ = [self.input_layer_] + self.hidden_layers_ + [self.output_layer_]

        self.init_weigth(activation)

    # prediccion de las clases cuantizada
    def predict(self, x):
        for layer in self.network_:
            x = layer.predict(x, 0)
        return numpy.where(x >= 0.5, 1, 0)

    # prediccion para los umbrales del semaforo

    def predict_ligthWare(self, x):
        ligths = [''] * len(x)

        for layer in self.network_:
            x = layer.predict(x, 0)

        for i in range(len(x)):
            if x[i] > 0.8:
                ligths[i] = 'red'
            if 0.8 > x[i] > 0.6:
                ligths[i] = 'orange'
            if 0.6 > x[i] > 0.4:
                ligths[i] = 'yellow'
            if 0.4 > x[i] > 0.2:
                ligths[i] = 'ligth-green'
            if 0.2 > x[i]:
                ligths[i] = 'green'

        return ligths
