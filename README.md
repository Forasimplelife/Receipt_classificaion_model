
# Image Classification with ResNet (use receipt of supermarket for an example)
This project implements an image classification model using a ResNet34 architecture. The dataset contains multiple classes, and the model is trained using PyTorch. The code includes data preprocessing, model training, evaluation, and debugging utilities for saving sample images and visualizing training progress.

(日本語)

（中国語）















# 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 精度 |
| :---: | :---: | :---: | :---: |
| MNIST-train | resnet50_mnist.pth | MNIST-test | 99.64% |

# 所需环境
torch==1.7.1
# 文件下载
## a.模型文件下载
训练所需的resnet50_mnist.pth可以在百度云或google drive下载。
**百度云链接：**
链接：https://pan.baidu.com/s/1apl5kspxGvjg4y6hLjSktQ?pwd=4mlv 
提取码：4mlv

**google drive 链接：**
[https://drive.google.com/file/d/1rFNsKgbUWKfp533Znsu0Jwz3XhQSxksM/view?usp=sharing](https://drive.google.com/file/d/1rFNsKgbUWKfp533Znsu0Jwz3XhQSxksM/view?usp=sharing)
## b.MNIST数据集下载
**百度云链接：**
链接: [https://pan.baidu.com/s/1MYMs_axknMm2g5Ou-cWmgQ](https://pan.baidu.com/s/1MYMs_axknMm2g5Ou-cWmgQ)
提取码: 8ce2 
# 预测步骤

1. 下载好预训练的模型或按照训练步骤训练好模型；
1. 在prediction.py文件里面，在如下部分修改PAHT使其对应训练好的模型路径；
```python
PATH = './logs/resnet50-mnist.pth'
```

3. 运行prediction.py，输入每次预测的图片个数。
# 训练步骤

1. 本文使用MNIST数据集进行训练，调用pytorch接口可以直接进行下载（代码已写好）；
1. 如果使用pytorch接口下载速度慢，可使用百度云进行下载。将下载后的文件放入data文件夹中即可；
1. 运行train.py即可开始训练。
# Reference
[https://github.com/bubbliiiing/faster-rcnn-pytorch.git](https://github.com/bubbliiiing/faster-rcnn-pytorch.git)
