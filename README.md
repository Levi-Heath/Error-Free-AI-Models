# Error-Free AI Models
Model weights for simple feedforward neural networks (FNNs) trained to 100% accuracy on the MNIST and FashionMNIST benchmarking datasets using the new training methods introduced by Deng in *Error-free Training for Artificial Neural Networks*.

## Models
| Dataset  | Model Architecture | Accuracy (%) |
| ------------- | ------------- | ------------- |
| MNIST  | 784-100-10 | 100.0 |
| FashionMNIST  | 784-100-10 | 100.0 |

## Training Method
These models were trained to a modest error rate using a standard stochastic gradient descent algorithm followed by a homotopy method, called gradient descent tunneling, to achieve errorless models. For a detailed explanation of gradient descent tunneling, read Deng's paper, which is available at https://arxiv.org/abs/2312.16060.


# Toward Errorless Training ImageNet-1k
We have also trained models using gradient descent tunneling on the ImageNet-1k dataset, which achieve a very high accuracy rate (98.3%). 
To learn more about these models, please read our manuscript, *Toward Errorless Training ImageNet-1k* (https://arxiv.org/abs/2508.04941).
The models and code that accompany this paper are hosted on the Hugging Face repository: https://huggingface.co/Levi-Heath/Towards-Errorless-Training-ImageNet-1k
