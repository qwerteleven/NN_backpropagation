import network
import backpropagation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def relu_derivate(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def relu_(x):
    x = np.clip(x, -5, 5)  # regularizacion
    x[x >= 0] = 0
    return x


def sigmoid_(x):
    x = np.clip(x, -5, 5)
    return 1 / (1 + np.exp(-x))

sigmoid = (  # funciones de activacion
    lambda x: sigmoid_(x),                      # x
    lambda x: sigmoid_(x) * (1 - sigmoid_(x))   # derivada(x)
)

relu = (
    lambda x: relu_(x),            # x
    lambda x: relu_derivate(x)    # derivada(x)
)

if __name__ == '__main__':
    learning_rate = 0.00002  # increiblemente sensible
    epoch = 100

    net = network.Network(learning_rate, epoch)

             #   input      hidden     output
    activations = [relu, relu, relu, relu, relu, sigmoid]  # activaciones de la red


    net.network_construct(input_shape=9, output_shape=1, hidden_layers=4, neurons_hidden_layers=[30, 50, 50, 30],
                          activation=activations)

    algoritm = backpropagation.BackPropagation()


    # load and prepare data
    filename = 'datos_BPNN.xls'

    dataset = pd.read_excel(filename)

    # eliminamos la serie temporal, ?¿ posible segundo trabajo con series temporales
    # elimino las variables no numericas -> posibilidad de one-hot-encoding,
    # para aumentar la explicacion del los datos

    dataset = dataset.drop(["FECHA_HORA", "INTENSIDAD_PRECIPITACION", "TIPO_PRECIPITACION", "ESTADO_CARRETERA"], axis=1)

    # plt.hist(dataset.iloc[:, 9])
    # plt.show()

    from sklearn.utils import resample
    #  balanceo de clases

    df_majority = dataset[dataset.iloc[:, 9] == 'No']
    df_minority = dataset[dataset.iloc[:, 9] == 'Yes']
    # Upsample minority class
    df_majority_downsampled = resample(df_majority,
                                     replace=False,  # sample with replacement
                                     n_samples=3500,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])


    # plt.hist(df_downsampled.iloc[:, 9])
    # plt.show()


    data = df_downsampled.values

    np.random.shuffle(data)   # randomiza los inputs


    x = data[:, :9]

    # reemplazo valores nulos en 0, ?¿ poner la media
    x = [[(i, 0)[i == ' '] for i in j] for j in x]

    x = np.asarray(x)

    y = data[:, 9]

    # convierto a binario las etiquetas 1 == accidente; 0 == no accidente
    y = [(1, 0)[i == "No"] for i in y]


    y = np.asarray(y)
    x = np.asarray(x)


    X_train = x[:int(len(x)/2), :]
    Y_train = y[:int(len(y)/2)].reshape(-1, 1)

    X_val = x[int(len(x)/2):, :]
    Y_val = y[int(len(y)/2):].reshape(-1, 1)

    error, accuracy, validation_error, validation_accuracy = algoritm.fit(X_train, Y_train, net, X_val, Y_val)

    epoch = list(range(0, epoch))
    plt.plot(epoch, error)
    plt.plot(epoch, accuracy)
    plt.plot(epoch, validation_error)
    plt.plot(epoch, validation_accuracy)
    plt.legend(['error', 'accuracy', 'validation_error', 'validation_accuracy'])
    plt.show()


    plt.hist(net.predict_ligthWare(X_val))  # umbrales para el semaforo
    plt.show()

    plt.hist(net.predict(X_train).ravel())
    plt.show()


    # print(net.predict(X_train).ravel())  # clases predichas


'''
    otro data set de prueba
    
   def circulo(num_datos=100, R=1, minimo=0, maximo=1):
        pi = math.pi
        r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
        theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10

        x = np.cos(theta) * r
        y = np.sin(theta) * r

        y = y.reshape((num_datos, 1))
        x = x.reshape((num_datos, 1))

        # We reduce the number of elements so that there is no overflow
        x = np.round(x, 3)
        y = np.round(y, 3)

        df = np.column_stack([x, y])
        return (df)


    datos_1 = circulo(num_datos=150, R=2)
    datos_2 = circulo(num_datos=150, R=0.5)
    X = np.concatenate([datos_1, datos_2])
    X = np.round(X, 3)

    Y = [0] * 150 + [1] * 150
    Y = np.array(Y).reshape(-1, 1)


    datos1 = circulo(num_datos=10, R=3)
    datos2 = circulo(num_datos=10, R=1)
    X_val = np.concatenate([datos1, datos2])
    X_val = np.round(X, 3)

    Y_val = [0] * 150 + [1] * 150
    Y_val = np.array(Y_val).reshape(-1, 1)

'''
