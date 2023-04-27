# 迁移学习的垃圾分类方法研究
# DeepLearning_GarbageClassification

## 1. 数据集
数据集已发布到[**GarbageSortingPictureDataSet**](https://github.com/GuoHuiTian/GarbageSortingPictureDataSet)仓库，需要的读者自行提取数据集。

数据集划分
> 训练集：验证集：测试集 = 8:1:1

| 样本类别 | 样本数量 |
| :--: | :--: | 
| 训练集 | 31550 |
| 验证集 | 3932 |
| 测试集 | 3866 |

## 2. 迁移学习实验
### 2.1 ResNet50模型
#### 2.1.1 ResNet模型介绍
ResNet也称为残差网络，VGGNet网络模型和GoogleLeNet模型都是通过增加网络的深度使网络获得了更好的性能，但是不可避免的是随着网络深度增加到一定程度会导致模型的产生梯度消失和梯度爆炸的问题，虽然这一问题可以通过归一化、初始化和BatchNormalization得到一定程度的解决。但是随着网络深度的增加，精度达到饱和，继续增加模型的深度，会导致精度的快速下降，致使模型退化。针对模型退化的问题，提出了残差网络模型（Residual Network），使用的是一种快捷连接方式（Shortcut Connection）。
ResNet残差块结构如图

![image](https://user-images.githubusercontent.com/131667281/234751388-7f36fd4c-91c1-4c78-a66b-f85b95f5a8c5.png)

结构中一共有两种映射关系，恒等映射（Identity Mapping）以及残差映射（Residual Mapping）。结构中的weight layer是普通的卷积操作, $F(x)$是残差，最后得到的输出为 $F(x)+x$，当网络到达 $F(x)+x$时，会通过前馈神经网络计算对比上一层的结果，如果网络模型达到最优解，则残差映射会变为0，剩下恒等映射，即上一层的最优解能够到达下一层的计算。如果 $F(x)+x$的前馈值优于 $x$，则会使用残差映射，去除恒等映射，理论上，模型会一直处于最优的结果，因此也就解决了随着网络深度的增大精度下降的问题。残差块的设计有两种方式，常规残差块中是两个 $3\times3$的卷积堆叠，瓶颈残差块是 $1\times1+3\times3+1\times1$的三层结构，这是由于常规残差块会有大量的计算，为了减少模型的计算量，因此对常规残差块做了优化得到瓶颈残差块。以一个残差快为例，常规残差块的计算量为 $3\times3+3\times3$，而瓶颈残差块的计算量仅为 $1\times1+3\times3+1\times1$。

#### 2.1.2 ResNet50迁移学习模型实验

