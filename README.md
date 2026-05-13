# ml
A hands-on collection of TensorFlow notebooks implementing core neural network architectures from scratch, applied to real-world datasets. Each notebook focuses on a single architecture paired with a dataset that naturally suits it.

## Physics-Informed Neural Networks (PINNs)
PINNs solving various non-linear partial differential equations:
goal $\downarrow$ time-domain $\rightarrow$ | continuous-time | discrete-time
-- | -- | --
solution | [Burgers](/PINN/Burgers.ipynb), [Schrodinger](/PINN/Schrodinger.ipynb) | [Burgers](/PINN/Burgers_RK.ipynb), [Allen-Cahn](/PINN/AllenCahn.ipynb)
discovery | [Navier-Stokes](/PINN/NavierStokes.ipynb) | Korteweg–de Vries

## Other notebooks
name | architecture | dataset
---- | ------------ | -------
mlp_iris | simple multi-layer perceptron (MLP) | iris flower
cnn_cifar10 | convolutional neural network (CNN) | CIFAR-10
wdn_higgs_boson | wide & deep network (WDN) | ATLAS's Higgs boson data
rnn_stocks | recurrent neural network (RNN) | NVIDIA stock price
lstm_stocks | long short-term memory (LSTM) RNN | NVIDIA stock price
