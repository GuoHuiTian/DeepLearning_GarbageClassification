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

![image](https://user-images.githubusercontent.com/131667281/234799631-6d2b092f-0352-49dd-9f3a-c08b2245a2be.png)

经过30个epoch后，模型在val上的loss值为`3.0818`，accuracy为`0.3082`，能够明显的看出，模型收敛的速度极其缓慢，为了参考模型的具体的准确率，重新设定`epoch=200`，实验结果如图

![image](https://user-images.githubusercontent.com/131667281/234800166-d4444f16-4451-451e-be2e-ae5d4c2ec40f.png)

可以看出模型并没有实质性的改变，因此还需要进一步优化模型，考虑ResNet50模型的迁移策略问题，在上述实验中，我们冻结了ResNet50模型的全部卷积层，只训练针对垃圾分类任务的全连接层，但是由于ResNet50模型的参数量较多，模型的层级较深，只训练全连接层并不能很好的获取垃圾分类任务的特征。紧接着我对另外的迁移学习策略进行了实验，分别冻结网络浅层的前80层，以及全部层级都设定为可训练，当然这里的epoch依旧设置为30次，分别得到下图的结果

![image](https://user-images.githubusercontent.com/131667281/234800317-a6e9ddd7-7ce3-4718-81db-05261ba1396b.png)

![image](https://user-images.githubusercontent.com/131667281/234800340-f420f009-f394-4813-ad00-693c2660636c.png)

针对于上述的实验结果能够明显看出，模型的收敛速度变快，并且准确率也得到了很大的提升。由于模型后期的应用问题，ResNet50模型博主就只研究到这里，读者可以根据实际任务需求对模型进行优化。

### 2.2 MobileNetV2模型
#### 2.2.1 MobileNetV2模型介绍
2017年，Google公司提出了MobileNet模型，并应于ImageNet数据集上，对比与VGG16模型，在网络参数减小30倍的情况下，准确率只比VGG16低0.9%，用于移动端和嵌入式设备里表现优秀。随之2018年，Google公司在MobileNet模型的基础上提出了MobileNetV2模型，进一步推动了轻量化模型的进步。

MobileNet模型是通过将标准卷积过程分离为深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）假设输入特征图的大小是 $D_F\times D_F\times M$，使用 $D_K\times D_K$的卷积核，输出的特征图的大小是 $D_H\times D_H\times N$，在这一层，使用标准卷积核的运算量为： $D_F\times D_F\times M\times N\times D_K\times D_K$，使用深度可分离卷积的计算量为： $D_F\times D_F\times M\times N+D_F\times D_F\times M\times D_K\times D_K$，由于提取过程中卷积核的改变，计算量有很大的减少，减少量为

$$
\frac{D_F\times D_F\times M\times N+D_F\times D_F\times M\times D_K\times D_K}{D_F\times D_F\times M\times N\times D_K\times D_K}=\frac{1}{N}+\frac{1}{D^2_K}
$$ 

MobileNetV2模型是针对于MobileNet模型的改进，在此基础上提出了逆向残差结构和线性瓶颈单元。上文中，我们介绍了ResNet的残差模块，逆向残差结构采用了与残差模块相反的设计方式，瓶颈残差模块的操作中，图像通道输入进来会通过逐点卷积后进行正常的卷积运算，然后在通过逐点卷积还原通道数。由于MobileNet中采用深度可分离卷积的方式，因此逆向残差模块中使用的方式如图

![image](https://user-images.githubusercontent.com/131667281/234794339-8c0b326a-95f9-43e3-b585-9dfdd54afebb.png)

线性瓶颈单元是MobileNetV2的另一个改进，当通过激活函数的通道数量较少时，ReLU函数很难保存完整的提取信息，因此MobileNetV2采用线性函数的方式，最大限度的保存提取的特征信息。MobileNetV2迁移学习模型如表

| Layer(type) | Output Shape | Param |
| :--: | :--: | :--: |
| rescaling | (None,224,224,3) | 0 |
| mobilenetv2_1.00_224 | (None,7,7,1280) | 2257984 |
| global_average_pooling2d | (None,1028) | 0 |
| dense | (None,120) | 153720 |

预训练模型的复用中，我们删除原始的分类器，加入针对于垃圾分类任务的分类器。并根据常用的三种迁移策略，对模型进行微调。

> 策略1：冻结全部卷积层，只训练针对垃圾分类任务的全连接层
> 策略2：冻结前140层网络，训练剩余卷积层以及全连接层
> 策略3：训练模型的全部卷积层以及全连接层

#### 2.2.2 实验结果
深度学习框架与超参数的设置跟ResNet50模型实验的设置相同。
三组不同的实验，模型的训练参数与冻结参数如表

| 模型 | 训练参数 | 冻结参数 |
| :--: | :--: | :--: |
| MobileNetV2_connect | 153720 | 2257984 |
| MobileNetV2_140after | 1193720 | 1217984 |
| MobileNetV2_all | 2377592 | 34112 |

**MobileNetV2_connect** 训练结果如图

![image](https://user-images.githubusercontent.com/131667281/234799308-72b3fabb-83e8-4367-ac8a-6c38f229cebc.png)

**MobileNetV2_140after** 训练结果如图

![image](https://user-images.githubusercontent.com/131667281/234798891-c63cea49-60c7-46e8-9fa0-b9fb5fa9722b.png)

**MobileNetV2_all** 训练结果如图

![image](https://user-images.githubusercontent.com/131667281/234798964-fc4b4934-6d46-4bb2-b2c8-3aa26fb4190f.png)

各模型的准确率如表

| 模型 | 验证集 | 测试集 |
| :--: | :--: | :--: |
| MobileNetV2_connect | 85.38% | 84.97% |
| MobileNetV2_140after | 86.51% | 85.31% |
| MobileNetV2_all | 82.45& | 81.76% |

经过对比实验，分析实验结果，在训练的参数方面，MobileNetV2_connect仅有MobileNetV2_all的6.5%，MobileNetV2_140after的训练参数约为MobileNetV2_all的1/2。预训练模型的迁移过程中，并不是训练的参数越多越好，由于预训练模型的一部分参数是固定的，因此实验的准确率更多看垃圾分类数据集与参数的拟合情况。根据图3-2、3-3、3-4的实验结果来看，MobileNetV2_connect是表现更加稳定，在30轮的训练中，模型并未出现抖动的情况，不管是准确率还是损失函数，稳定性都是最好的，而MobileNetV2_140after模型，虽然准确率的表现整体上比MobileNetV2_connect更好，但是由于损失函数不稳定，出现了抖动等情况。垃圾分类的实际应用中，模型的稳定性也是一个重要因素，模型的不稳定容易导致垃圾分类错误，投放到错误的垃圾类别中，造成资源的浪费等。对于MobileNetV2_all模型来说，虽然训练的参数最多，但是由于整体的准确率不如其他两个模型，这是因为训练的参数较多，但数据集的图片数量并不能有效提供给模型进行学习，且数据分类较多，难以达到很高的识别准确率，然而扩充数据集是一项复杂的工程，且在实际的分类任务中，数据集的大小很难达到深度学习模型的要求，并不适合应用与此次垃圾分类任务中。

#### 2.2.3 模型优化
通过上述实验，最终确定MobileNetV2_connect作为我们主要的研究模型。MobileNetV2_connect模型在训练集和验证集上出现了明显的差异，训练集的准确率为99.53%，而验证集仅有85.38%，这说明模型存在过拟合问题。主要为在训练集上的拟合度过高，导致模型在应用中如果使用与训练集相仿的图片时识别准确率很高，但是将同一垃圾类型图片风格改变会导致模型的准确率降低，因此，需要对模型进行一定的优化。
防止过拟合的主要机制有数据增强、正则化、随机权重初始化、Dropout等。在MobileNetV2_connect模型中，由于我们只是训练全连接层，垃圾分类任务下输入全连接层的所有神经元均被拟合进模型中，模型在训练集上才会出现过拟合情况，因此在冻结全部卷积层的实验策略里，我们首要考虑丢弃一部分全连接层前的神经元，即在全连接层前加入Dropout机制，以0.5的概率丢弃一部分神经元，以达到防止模型过拟合的效果，加入Dropout机制后MobileNetV2_connect的训练结果如图

![image](https://user-images.githubusercontent.com/131667281/234801924-1dcd418a-e5fa-4539-98e9-e6b9ab04febd.png)

模型的准确率、loss函数的大小如表

|  | Train | Validation | Test |
|:--:| :--: | :--: | :--: |
| Accuracy | 89.57% | 85.38% | 85.28% |
| Loss | 0.3522 | 0.5315 | 0.5748 |

在加入Dropout机制后，模型的表现稳定，且准确率依旧能够保持85%，因此将该模型确定为实际垃圾分类任务中的模型。

