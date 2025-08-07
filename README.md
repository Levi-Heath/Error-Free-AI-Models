# Error-Free AI Models
Model weights for simple artificial neural networks (ANNs) that are trained to or near 100% accuracy on several benchmarking datasets. 

## Models
| Dataset  | Model Architecture | Accuracy |
| ------------- | ------------- | ------------- |
| MNIST  | 784-100-10 feedforward neural network (FNN)  | 100.0% |
| FashionMNIST  | 784-100-10 FNN  | 100.0% |
| ImageNet-1k | 17-40-2 Feature Model with 900-256-25 FNNs |  98.3% |

### Description of the ImageNet-1k Feature Model
For a thorough description of our models trained on the ImageNet-1k dataset, please read our preprint, "Towards Errorless Training ImageNet-1k", which
 is available at [ADD LINK to arXiv preprint].
 In ../ImageNet-1k/MATLAB, we give parameters for 6 models, which are listed in the table below. Each model has the following architecture: 
 17x40x2=1360 FNNs, all with homogeneous architecture (900-256-25 or 900-256-77-25), working in parrallel to produce 1360 predictions which 
 determine a final prediction using the majority voting protocol. We trained models using the 64x64 downsampled ImageNet-1k dataset. We transformed the data by...[ADD DESCRIPTION OF TRANSFORMATION].

| Model | Training Method | FNN Architecture |
| ------------- | ------------- | ------------- |
| Model_S_h1_m1 | SGD | 900-256-25 |
| Model_S_h1_m2 | SGD | 900-256-25 |
| Model_S_h2_m1 | SGD | 900-256-77-25 |
| Model_T_h1_m1 | SGD followed by GDT | 900-256-25 |
| Model_T_h1_m2 | SGD followed by GDT | 900-256-25 |
| Model_T_h2_m1 | SGD followed by GDT | 900-256-77-25 |
*SGD = stochastic gradient descent
**GDT = gradient descent tunneling