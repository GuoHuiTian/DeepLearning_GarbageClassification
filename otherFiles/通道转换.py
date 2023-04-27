from PIL import Image
import os


path1="F:/毕设_垃圾分类/总结综合/images/"

file1=os.listdir(path1)

for i in file1:
    path2=path1+i
    file2=os.listdir(path2)
    for j in file2:
        path3=path2+"/"+j
        print(path3)
        path = path3
        all_images = os.listdir(path)
        # print(all_images)

        for image in all_images:
            image_path = os.path.join(path, image)
            img = Image.open(image_path)  # 打开图片
            img = img.convert("RGB")  # 4通道转化为rgb三通道
            save_path = path
            img.save(save_path + image)
