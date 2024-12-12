
# **ResNetを使用した画像分類（スーパーの領収書を例として）**
本プロジェクトでは、ResNetアーキテクチャを使用した画像分類モデルを実装します。データセットには複数のクラスが含まれており、PyTorchを使用してモデルを訓練します。手順には、データの前処理、モデルの訓練、評価、デバッグ用のユーティリティ（サンプル画像の保存や訓練進捗の可視化）が含まれます。

## **Table of Contents**
- [プロジェクト概要](#プロジェクト概要)
- [データセット](#データセット)
- [モデルアーキテクチャー](#デルアーキテクチャー)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [トレーニングと評価](#トレーニングと評価)
- [結果](#結果)
- [貢献](#貢献)
- [ライセンス](#ライセンス)


## **プロジェクト概要**

This project demonstrates image classification using a deep learning model based on the **ResNet34** architecture. The key components include:

- **Data Augmentation and Preprocessing**: Transformations applied to input images, including resizing and normalization.
- **Model Training and Validation**: Includes loss computation, backpropagation, and optimizer updates.
- **Debugging Tools**: Functions to save and visualize sample images during training.
- **Plotting Functions**: For visualizing random samples from each class during data exploration.

## **Dataset prepare**

I prepared the two datasets named as train and val

 <img width="420" height="200" src=figure/1.png/> 

Images in the foler look like 

 <img src="figure/2.png", width="100%"> 


## **Model Architecture**

This project uses **ResNet34** as the backbone architecture for image classification. Key features of ResNet include:
- **Residual Blocks**: Allowing the model to train deeper layers by skipping connections.
- **Convolutional Layers**: Used to extract features from images.
- **Fully Connected Layer**: Used for classification into different categories.


## **Usage**

### 1.Data Preprocessing

* [dataloader.py](datamodule/dataloader.py)  

The dataset is loaded using the `RotatedReceiptDataset` class, which is defined in the `dataset_module`.

The data is preprocessed using the following transformations:
- **Resize**: All images are resized to 224x224 pixels.
- **Normalization**: Images are normalized to `[0.5, 0.5, 0.5]` for each RGB channel.

I use the dataloader to rotated images and preparing other 3 classes, then dataset used consists of four classes of images:
1. Raw images (Class 1)
2. Rotated images (90° left, Class 2)
3. Rotated images (180° left, Class 3)
4. Rotated images (90° right, Class 4)

 <img width="800" height="500" src=figure/3.png/> 

### 2. training and evaluation
* [data_load_train.py](data_load_train.py)  

### 3. prediction
* [prediction.py](prediction.py)  

### results


### Contribution

### 
