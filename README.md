# Error-Free AI Models
Model weights for simple artificial neural networks (ANNs) that are trained to or near 100% accuracy on several benchmarking datasets. 

## Models
| Dataset  | Model Architecture | Accuracy (%) |
| ------------- | ------------- | ------------- |
| MNIST  | 784-100-10 feedforward neural network (FNN)  | 100.000 |
| FashionMNIST  | 784-100-10 FNN  | 100.000 |
| ImageNet-1k | 17-40-2 Feature Model with 900-256-25 FNNs |  98.300 |

### Description of the ImageNet-1k Feature Model
For a thorough description of our models trained on the ImageNet-1k dataset, please read our preprint, *Towards Errorless Training ImageNet-1k*, which is available at [ADD LINK to arXiv preprint]. In ../ImageNet-1k/MATLAB, we give parameters for 6 models, which are listed in the table below. Each model has the following architecture: $17\times 40\times 2=1360$ FNNs, all with homogeneous architecture (900-256-25 or 900-256-77-25), working in parrallel to produce 1360 predictions which determine a final prediction using the majority voting protocol. We trained models using the following transformation of the $64\times 64$ downsampled ImageNet-1k dataset:
 - downsampled images to $32\times 32$, using the mean values of non-overlapping $2\times 2$ grid cells and
 - trimmed off top row, bottom row, left-most column, and right-most column.

 This transformed data results in $30\times 30$ images, hence 900-dimensional input vectors.


| Model | Training Method | FNN Architecture | Accuracy (%) |
| ------------- | ------------- | ------------- | ------------- |
| Model_S_h1_m1 | SGD | 900-256-25 | 98.247 |
| Model_S_h1_m2 | SGD | 900-256-25 | 98.299 |
| Model_S_h2_m1 | SGD | 900-256-77-25 | 96.990 |
| Model_T_h1_m1 | SGD followed by GDT | 900-256-25 | 98.289 |
| Model_T_h1_m2 | SGD followed by GDT | 900-256-25 | 98.300 |
| Model_T_h2_m1 | SGD followed by GDT | 900-256-77-25 | 97.770 |
*SGD = stochastic gradient descent
**GDT = gradient descent tunneling