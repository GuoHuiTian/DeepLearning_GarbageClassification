# -*- coding: UTF-8 –*-
import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil



class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('智能垃圾分类系统')
        self.model = tf.keras.models.load_model("models/Dropout_MobileNet_epoch30.h5")  # 修改为自己的模型路径
        self.to_predict_name = "images/Init.jpeg"
        self.class_names = ['其他垃圾_PE塑料袋', '其他垃圾_一次性杯子', '其他垃圾_一次性棉签', '其他垃圾_口罩', '其他垃圾_唱片', '其他垃圾_干燥剂', '其他垃圾_打火机', '其他垃圾_搓澡巾', '其他垃圾_滚筒纸', '其他垃圾_点燃的香烟', '其他垃圾_眼镜', '其他垃圾_笔', '其他垃圾_胶带', '其他垃圾_验孕棒', '其他垃圾_鸡毛掸', '厨余垃圾_圣女果', '厨余垃圾_地瓜', '厨余垃圾_大蒜', '厨余垃圾_梨', '厨余垃圾_橙子', '厨余垃圾_火龙果', '厨余垃圾_瓜子壳', '厨余垃圾_番茄', '厨余垃圾_白菜叶', '厨余垃圾_苹果', '厨余垃圾_草莓', '厨余垃圾_菠萝', '厨余垃圾_菠萝蜜', '厨余垃圾_蘑菇', '厨余垃圾_蛋', '厨余垃圾_蛋挞', '厨余垃圾_蛋糕', '厨余垃圾_西瓜皮', '厨余垃圾_豆', '厨余垃圾_豆腐', '厨余垃圾_辣椒', '厨余垃圾_面包', '厨余垃圾_饼干', '厨余垃圾_香蕉皮', '可回收物_A4纸', '可回收物_一次性筷子', '可回收物_不锈钢管', '可回收物_乒乓球拍', '可回收物_书', '可回收物_保温杯', '可回收物_信封', '可回收物_充电头', '可回收物_充电宝', '可回收物_充电线', '可回收物_凳子', '可回收物_刀', '可回收物_剪刀', '可回收物_勺子叉子', '可回收物_包', '可回收物_卡', '可回收物_台灯', '可回收物_吹风机', '可回收物_呼啦圈', '可回收物_塑料瓶', '可回收物_塑料盆', '可回收物_头戴式耳机', '可回收物_尺子', '可回收物_布制品', '可回收物_帽子', '可回收物_手机', '可回收物_手电筒', '可回收物_手表', '可回收物_打气筒', '可回收物_拉杆箱', '可回收物_插线板', '可回收物_收音机', '可回收物_放大镜', '可回收物_易拉罐', '可回收物_望远镜', '可回收物_木制切菜板', '可回收物_木质梳子', '可回收物_木质锅铲', '可回收物_枕头', '可回收物_桌子', '可回收物_水壶', '可回收物_水杯', '可回收物_泡沫板', '可回收物_灭火器', '可回收物_热水瓶', '可回收物_电动剃须刀', '可回收物_电子秤', '可回收物_电熨斗', '可回收物_电磁炉', '可回收物_电路板', '可回收物_电风扇', '可回收物_电饭煲', '可回收物_盘子', '可回收物_碗', '可回收物_箱子', '可回收物_纸板', '可回收物_衣架', '可回收物_袜子', '可回收物_裙子', '可回收物_裤子', '可回收物_计算器', '可回收物_订书机', '可回收物_话筒', '可回收物_路由器', '可回收物_轮胎', '可回收物_遥控器', '可回收物_钥匙', '可回收物_铁丝球', '可回收物_键盘', '可回收物_镊子', '可回收物_闹钟', '可回收物_雨伞', '可回收物_鞋', '可回收物_鼠标', '有害垃圾_太阳能电池', '有害垃圾_灯', '有害垃圾_电池', '有害垃圾_纽扣电池', '有害垃圾_胶水', '有害垃圾_药品包装', '有害垃圾_蓄电池']
        self.resize(1080, 800)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 18)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("识别样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images_test/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images_test/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images_test/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' 垃圾种类 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))

        label_result_f = QLabel(' 物品名称 ')
        self.result_f = QLabel("等待识别")
        label_result_g=QLabel('概率大小')
        self.result_g=QLabel("等待识别")

        self.label_info = QTextEdit()
        self.label_info.setFont(QFont('楷体', 12))

        label_result_f.setFont(QFont('楷体', 16))
        self.result_f.setFont(QFont('楷体', 24))

        label_result_g.setFont(QFont('楷体',16))
        self.result_g.setFont(QFont('楷体',20))

        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_g, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_g, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.label_info, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)


        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '首页')
        self.setTabIcon(0, QIcon('images/首页.jpg'))


    # 上传图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.jpg *.png *jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            target_image_name = "images_test/tmp_single" + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images_test/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images_test/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images_test/show.png"))

    # 预测图片
    def predict_img(self):
        img = Image.open('images_test/target.png')
        img = np.asarray(img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        result1=np.squeeze(outputs)
        result_index = int(np.argmax(outputs))
        Prob=result1[result_index]
        result = self.class_names[result_index]
        names = result.split("_")
        # print(result)
        if names[0] == "厨余垃圾":
            self.label_info.setText(
                "厨余垃圾包括剩菜剩饭、骨头、菜根菜叶、果皮等食品类废物。经生物技术就地处理堆肥，每吨可生产0.6~0.7吨有机肥料。")
        if names[0] == "有害垃圾":
            self.label_info.setText(
                "有害垃圾含有对人体健康有害的重金属、有毒的物质或者对环境造成现实危害或者潜在危害的废弃物。包括电池、荧光灯管、灯泡、水银温度计、油漆桶、部分家电、过期药品及其容器、过期化妆品等。这些垃圾一般使用单独回收或填埋处理。")
        if names[0] == "可回收物":
            self.label_info.setText(
                " 根据《城市生活垃圾分类及其评价标准》行业标准以及参考德国垃圾分类法，可回收物是指适宜回收循环使用和资源利用的废物。主要包括：纸类，塑料，金属，玻璃，织物等。主要的处理方式有：1.垃圾再生法；2.垃圾焚烧法；3.垃圾堆肥法；4.垃圾生物降解法。")
        if names[0] == "其他垃圾":
            self.label_info.setText(
                "其他垃圾指危害比较小，没有再次利用的价值的垃圾，其他垃圾包括砖瓦陶瓷、渣土、卫生间废纸、瓷器碎片、动物排泄物、一次性用品等难以回收的废弃物。一般都采取填埋、焚烧、卫生分解等方法处理，部分还可以使用生物分解的方法解决")
        self.result.setText(names[0])
        self.result_f.setText(names[1])
        self.result_g.setText(str(Prob))

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
