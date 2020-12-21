import numpy as np
import cv2
import os
import random
from tqdm import tqdm
# calculate means and std
train_txt_path = 'C:/Users/94806/Desktop/list.txt'

CNum = 200  # 挑选多少图片进行计算

img_h, img_w = 1200, 1200
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    #print(lines)
    #normMean = [0.4752129, 0.4600728, 0.4488067]
    #normStd = [0.27147016, 0.27277786, 0.27487788]
    #normMean = [0.48610604, 0.47190934, 0.46161118]
    #normStd = [0.28547028, 0.2868505, 0.2889218]

    #normMean = [0.46381295, 0.45201838, 0.4413544]
    #normStd = [0.27268183, 0.26858312, 0.27312627]
    #normMean = [0.4608172, 0.44619986, 0.43740782]
    # normStd = [0.2696413, 0.2728617, 0.27502558]
    random.shuffle(lines)  # shuffle , 随机挑选图片

    for i in tqdm(range(CNum)):
        img_path = lines[i].split('\n')[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=3)
#         print(i)

imgs = imgs.astype(np.float32) / 255.

for i in tqdm(range(3)):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))