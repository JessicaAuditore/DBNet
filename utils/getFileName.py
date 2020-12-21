import os


def get_imlist(path):
    a = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    return a


img_path = get_imlist("D:/develop/dataset/ICDAR2017RCTW/train")

file = open('C:/Users/94806/Desktop/list.txt', 'w')
for i, name in enumerate(img_path):
    name = name.replace("\\", "/")
    file.write(name + '\n')
