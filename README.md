# Error-Free AI Models
Model weights for simple feedforward neural networks (FNNs) trained to 100% accuracy on the MNIST and FashionMNIST benchmarking datasets using the new training methods introduced by Deng in *Error-free Training for Artificial Neural Networks*.

## Models
| Dataset  | Model Architecture | Accuracy (%) |
| ------------- | ------------- | ------------- |
| MNIST  | 784-100-10 | 100.0 |
| FashionMNIST  | 784-100-10 | 100.0 |

## Training Method
These models were trained to a modest error rate using a standard stochastic gradient descent algorithm followed by a homotopy method, called gradient descent tunneling, to achieve errorless models. For a detailed explanation of gradient descent tunneling, read Deng's paper, which is available at https://arxiv.org/abs/2312.16060.
