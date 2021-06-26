import numpy


class BackPropagation(object):
    """
    Class BackPropagation:

       Attributes:
         sse_while_fit_.-

       Methods:
         fit(p_X_training, p_Y_training, p_X_validation, p_Y_validation, network)
         predict(p_x) .- Method to predict the output, y

    """

    def fit(self, train_x, train_y, network, test_x=None, test_y=None):

        error = []
        accuracy = []

        validation_error = []
        validation_accuracy = []

        for epoch in range(network.epoch):
            if epoch % 20 == 0:
                print("Epoch:", epoch)

            a_s = self.forward(train_x, network)
            train_predict = self.backpropagation(train_y, a_s, network)
            temp = self.mse(numpy.round(train_predict), train_y)
            acc = self.accuracy(numpy.round(train_predict), train_y)
            error.append(temp)
            accuracy.append(acc)

            if test_x is not None and test_y is not None:
                a_v = self.forward(test_x, network)
                val_predict = self.backpropagation_lock_weigth(test_y, a_v, network)  # dont learn
                val_error = self.mse(numpy.round(val_predict), test_y)
                val_acc = self.accuracy(numpy.round(val_predict), test_y)
                validation_error.append(val_error)
                validation_accuracy.append(val_acc)

        return error, accuracy, validation_error, validation_accuracy

    def forward(self, x, network):

        a_s = [x]

        for layer in network.network_:
            a = layer.predict(a_s[-1], 0)
            a_s.append(a)

        return a_s

    def mse(self, output, target):
        x = (numpy.array(target) - numpy.array(output)) ** 2

        return numpy.mean(x)

    def mse_derivate(self, output, target):
        x = (numpy.array(target) - numpy.array(output))

        return -x

    def sigmoid(self):
        return lambda x: 1 / (1 + numpy.exp(-x))

    def binary_cross_entropy(self, output, target):

        x = (target * -numpy.log(self.sigmoid()(output)) +
             (1 - target) * -numpy.log(1 - self.sigmoid()(output)))

        return numpy.mean(x)

    def binary_cross_entropy_derivate(self, output, target):

        x = (target * -numpy.log(self.sigmoid()(output)) +
             (1 - target) * -numpy.log(1 - self.sigmoid()(output)))

        return x

    def accuracy(self, output, target):
        examples = len(output)
        acc = 0
        for i in range(examples):
            if output[i] == target[i]:
                acc += 1

        return acc / examples

    def backpropagation_lock_weigth(self, Y, a_s, network):

        delta = []  # errors

        for index, layer in reversed(list(enumerate(network.network_))):
            a = a_s[index + 1]

            if index == len(network.network_) - 1:
                x = self.mse_derivate(a, Y) * layer.activation[1](a)
                delta.append(x)

            else:
                x = numpy.matmul(delta[-1], w_) * layer.activation[1](a)
                delta.append(x)

            w_ = layer.w[1:, :].T

        return a_s[-1]

    def backpropagation(self, Y, a_s, network):

        delta = []  # errors
        for index, layer in reversed(list(enumerate(network.network_))):

            a = a_s[index + 1]

            if index == len(network.network_) - 1:
                x = self.mse_derivate(a, Y) * layer.activation[1](a)
                delta.append(x)

            else:
                x = numpy.matmul(delta[-1], w_) * layer.activation[1](a)
                delta.append(x)

            w_ = layer.w[1:, :].T

            layer.w[0, :]  = layer.w[0, :] - numpy.mean(delta[-1], axis=0, keepdims=True) * network.eta
            layer.w[1:, :] = layer.w[1:, :] - numpy.matmul(a_s[index].T, delta[-1]) * network.eta


        return a_s[-1]
