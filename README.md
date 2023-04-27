# 迁移学习的垃圾分类方法研究

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
深度学习任务中，需要大量的原始数据集进行模型的训练，实际应用中，数据集的大小很难真正达到深度学习的要求，因此迁移学习能够很好地应用到新领域。ResNet50预训练模型已经在ImageNet数据集上经过训练，并且取得不错的识别效果。实验基于ResNet50预训练模型，迁移过程中删除原始全连接层，制定针对垃圾分类的训练任务。ImageNet数据集参数提取后，冻结所有层的参数，在全连接层前加入像素归一化、全局平均池化操作，设定垃圾分类全连接层分类器。
迁移学习的ResNet50模型结构如表

| Layer(type) | Output Shape | Param |
| :--: | :--: | :--: |
| rescaling | (None,224,224,3) | 0 |
| resnet50 | (None,7,7,2048) | 23587712|
| global_average_pooling2d | (None,2048) | 0 |
| dense | (None,120) | 245880 |


#### 2.1.3 实验结果
深度学习框架使用**TensorFlow**，超参数的设定如表

| 参数名 | 参数值 |
| :--: | :--: |
| image_size | $224\times224\times3$ |
| batch_size | 16 |
| optimizers | Adam |
| learning_rate | 0.0001 |
| loss | categorical_crossentropy |
| epochs | 30 |

模型训练的结果如图

![原始_ResNet50_epoch30](https://user-images.githubusercontent.com/131667281/234781749-848f9703-f9b1-48bc-9aa1-2edb89897841.png)

经过30个epoch后，模型在val上的loss值为`3.0818`，accuracy为`0.3082`，能够明显的看出，模型收敛的速度极其缓慢，为了参考模型的具体的准确率，重新设定`epoch=200`，实验结果如图

![原始_ResNet50_epoch200](https://user-images.githubusercontent.com/131667281/234782891-6e0c7a1b-afaa-4cd6-a78e-f9eee1251921.png)

可以看出模型并没有实质性的改变，因此还需要进一步优化模型，考虑ResNet模型的迁移策略问题，在上述实验中，我们冻结了ResNet50模型的全部卷积层，只训练针对垃圾分类任务的全连接层，但是由于ResNet50模型的参数量较多，模型的层级较深，只训练全连接层并不能很好的获取垃圾分类任务的特征。紧接着我对另外的迁移学习策略进行了实验，分别冻结网络浅层的前80层，以及全部层级都设定为可训练，当然这里的epoch依旧设置为30次，分别得到下图的结果

![ResNet50_80layers_epoch30](https://user-images.githubusercontent.com/131667281/234784474-2b0ec96a-4973-4f51-8dfd-f0bac9d32f91.png)

![ResNet50_all_layers_epoch30](https://user-images.githubusercontent.com/131667281/234784511-7f5b2a19-c2e7-4b3c-b23e-b754211daefc.png)
