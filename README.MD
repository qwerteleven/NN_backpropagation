# Neural Network — Backpropagation

A from-scratch implementation of a fully connected [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) with [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in Python, applied to road accident risk prediction. No deep learning frameworks are used — only NumPy.

---

## Architecture

The network is built from composable layer classes sharing a common abstract base:

```
Input → HiddenLayer × N → Output
```

| Class | Role |
|---|---|
| `Layer` (abstract) | Defines `_net_input()` (Xw + b) and the layer interface |
| `InputLayer` | Identity pass-through; weights initialised as the identity matrix |
| `HiddenLayer` | Fully connected; weights initialised from N(0, 0.5) |
| `OutputLayer` | Fully connected; same initialisation as hidden layers |
| `Network` | Assembles layers, manages weight init and activation assignment |
| `BackPropagation` | Training loop: forward pass, gradient computation, weight update |

### Default Configuration (`main.py`)

```
Input (9) → Dense(30) → Dense(50) → Dense(50) → Dense(30) → Output(1)
Activations: ReLU × 5, Sigmoid (output)
Learning rate: 0.00002
Epochs: 100
```

---

## How It Works

### Forward Pass

Each layer computes `Z = X·W + b` via `_net_input()`, then applies its activation function. The full set of intermediate activations `a_s` is stored for use in backpropagation.

### Backpropagation

The `backpropagation()` method iterates through layers in reverse, computing the [gradient](https://en.wikipedia.org/wiki/Gradient) of the [MSE loss](https://en.wikipedia.org/wiki/Mean_squared_error) at each layer using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule):

```
δ_output = MSE'(a, Y) ⊙ activation'(a)
δ_hidden = (δ_next · W^T) ⊙ activation'(a)
```

Weights and biases are then updated by [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent):

```
W ← W − η · (a_prev^T · δ)
b ← b − η · mean(δ)
```

A separate `backpropagation_lock_weight()` method runs the same computation on validation data without updating weights, allowing honest validation metrics each epoch.

### Activation Functions

| Function | Used in | Notes |
|---|---|---|
| [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) | Hidden layers | Clipped to [−5, 5] for regularisation |
| [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) | Output layer | Clipped to [−5, 5]; output ∈ (0, 1) |

Both are defined as `(f, f')` tuples so the same layer code handles both forward and backward passes.

### Traffic Light Output — `predict_lightWare()`

Beyond binary prediction, the network outputs a **risk level** mapped to five colour-coded bands:

| Output probability | Risk level |
|---|---|
| > 0.8 | 🔴 Red |
| 0.6 – 0.8 | 🟠 Orange |
| 0.4 – 0.6 | 🟡 Yellow |
| 0.2 – 0.4 | 🟢 Light green |
| < 0.2 | 🟢 Green |

---

## Dataset

### Road Accident Data (`main.py`)

The primary application predicts road accident likelihood (`Yes`/`No`) from 9 numeric road condition features (timestamp, precipitation type, road state columns are dropped). The dataset is [class-balanced](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) by downsampling the majority class to 3,500 samples using scikit-learn's `resample`. Labels are binarised: `1 = accident, 0 = no accident`.

### Seeds Dataset (`seeds_dataset.csv`)

A secondary dataset with 210 samples of wheat kernel geometry (area, perimeter, compactness, etc.) across 3 varieties. Used for standalone experimentation.

### Sine Approximation (`pruebas2.py`)

A scratch script that trains a two-hidden-layer network to approximate `sin(x)` over [0, 2π] using only NumPy, serving as a minimal proof-of-concept for the backpropagation logic.

---

## Project Structure

```
nn_backpropagation/
├── layer.py              # Abstract base class for all layers
├── input_layer.py        # Identity input layer
├── hidden_layer.py       # Fully connected hidden layer
├── output_layer.py       # Fully connected output layer
├── network.py            # Network assembly and prediction
├── backpropagation.py    # Training loop and gradient computation
├── main.py               # Data loading, training, plotting (accident dataset)
├── pruebas2.py           # Minimal sine approximation experiment
└── seeds_dataset.csv     # Wheat kernel geometry dataset
```

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

The primary training script (`main.py`) also requires an Excel file `datos_BPNN.xls` with road condition data (not included).

---

## License

MIT