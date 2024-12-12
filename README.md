
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

このプロジェクトは、**ResNet34** アーキテクチャに基づいたディープラーニングモデルを使用した画像分類をデモンストレーションするものです。主な構成要素は以下の通りです
This project demonstrates image classification using a deep learning model based on the **ResNet34** architecture. The key components include:

**データ拡張と前処理**：入力画像に適用される変換処理（リサイズや正規化など）。
**モデルのトレーニングと検証** ：損失計算、逆伝播、オプティマイザの更新を含む処理。
**デバッグツール**：トレーニング中にサンプル画像を保存および可視化する機能。
**プロット機能**：データ探索中に各クラスからランダムに選んだサンプルを可視化する機能。

## **データセットを準備**

Train と　Valを二つのフォルダーを作ります。

 <img width="420" height="200" src=figure/1.png/> 


<div align="medium">
  <img src="figure/2.png", width="100%"> 
</div>

## **モデルアーキテクチャー**

このプロジェクトでは、画像分類のバックボーンアーキテクチャとして**ResNet3**4を使用しています。ResNetの主な特徴は以下の通りです：
**Residual Blocks 残差ブロック**：スキップ接続を活用し、より深い層をトレーニング可能にする。
**Convolutional Layers 畳み込み層**：画像から特徴を抽出するために使用される。
**Fully Connected Layer 全結合層**：異なるカテゴリに分類するために使用される。


## **使用方法**

### 1.データ前処理

* [dataloader.py](datamodule/dataloader.py)  

このデータセットは、dataset_module内で定義されているRotatedReceiptDatasetクラスを使用して読み込まれます。

データは以下の変換処理を用いて前処理されます：
**Resize リサイズ**：すべての画像を224x224ピクセルにリサイズ。
**Normalization　正規化**：各RGBチャンネルごとに[0.5, 0.5, 0.5]に正規化。

データローダーを使用して回転画像を準備し、以下の4つのクラスで構成されるデータセットを作成します：
	1.	元の画像（クラス1）
	2.	左に90°回転した画像（クラス2）
	3.	左に180°回転した画像（クラス3）
	4.	右に90°回転した画像（クラス4）

 <img width="800" height="500" src=figure/3.png/> 

### 2. training and evaluation
* [data_load_train.py](data_load_train.py)  

### 3. prediction
* [prediction.py](prediction.py)  

### results


### Contribution

### 
