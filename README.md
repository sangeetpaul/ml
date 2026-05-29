# ml
A hands-on collection of TensorFlow notebooks implementing core neural network architectures from scratch, applied to real-world datasets. Each notebook focuses on a single architecture paired with a dataset that naturally suits it.

## Physics-informed machine learning
### Physics-informed neural networks (PINNs)
PINNs solving various non-linear partial differential equations (PDEs):
goal $\downarrow$ | continuous-time | discrete-time
-- | -- | --
solution | [Burgers](/PINN/Burgers.ipynb), [Schrodinger](/PINN/Schrodinger.ipynb) | [Burgers](/PINN/Burgers_RK.ipynb), [Allen-Cahn](/PINN/AllenCahn.ipynb)
discovery | [Navier-Stokes](/PINN/NavierStokes.ipynb) | [Korteweg–de Vries](/PINN/KdV.ipynb)

### Other physics-informed approaches
notebook | architecture | dataset
-------- | ------------ | -------
[PDET](/PDET.ipynb) | PDE Transformer | Kolmogorov flow
<!-- [FNO](/FNO.ipynb) | Fourier neural pperator (FNO) | Burgers equation -->

## Other approaches
notebook | architecture | dataset
-------- | ------------ | -------
[MLP_iris](/MLP_iris.ipynb) | simple multi-layer perceptron (MLP) | iris flower
[CNN_cifar10](/CNN_cifar10.ipynb) | convolutional neural network (CNN) | CIFAR-10
[WDN_higgs_boson](/WDN_higgs_boson.ipynb) | wide & deep network (WDN) | ATLAS's Higgs boson data
[RNN_stocks](/RNN_stocks.ipynb) | recurrent neural network (RNN) | NVIDIA stock price
[LSTM_stocks](/LSTM_stocks.ipynb) | long short-term memory (LSTM) RNN | NVIDIA stock price
