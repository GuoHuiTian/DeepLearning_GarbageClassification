import tensorflow as tf
import matplotlib.pyplot as plt
from time import *

image_path ="file_path"  # 代码所用图集的文件夹
train_dir = image_path  + "train"
validation_dir = image_path  + "val"


img_height = 224
img_width = 224
batch_size = 32


def data_load(train_dir, validation_dir,img_height, img_width, batch_size):
    train_ds1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,#压缩像素同时进行随机水平反转
                                           horizontal_flip=True,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           )
    train_ds2=train_ds1.flow_from_directory(directory=train_dir,#训练集目录
                                batch_size=batch_size,#每一批图像数据的数目
                                shuffle=True,#是否随机打乱
                                target_size=(img_height, img_width),#输入网络的尺寸大小
                                class_mode='categorical')#分类的方式)

    val_ds1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_ds2=val_ds1.flow_from_directory(directory=validation_dir,#训练集目录
                                batch_size=batch_size,#每一批图像数据的数目
                                shuffle=True,#是否随机打乱
                                target_size=(img_height, img_width),#输入网络的尺寸大小
                                class_mode='categorical')#分类的方式)

    class_names = train_ds2.class_indices
    print(class_names)
    return train_ds2, val_ds2, class_names

data_load(train_dir, validation_dir,img_height, img_width, batch_size)
