import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import pandas as pd
import keras

from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras import backend as K
from tqdm import tqdm
from keras.preprocessing.image import load_img

# 导入训练数据集
y_train = pd.read_excel("train.xlsx")
train_image_dir = "./train/"
# 将标签单独取出，转成独热码
breed = pd.Series(y_train['breed'])
one_hot = pd.get_dummies(breed, sparse = True)
one_hot_labels = np.asarray(one_hot)
# 读取训练数据集文件名（train.xlsx的第一列）
trainFilenames=pd.read_excel("train.xlsx", usecols=0)
# 转成numpy矩阵，再转成列表，取出文件名，组合成路径。...
trainFilenames=trainFilenames.T.as_matrix()
trainFilenames=trainFilenames.tolist()
trainFilenames = [y for x in trainFilenames for y in x]
train_img_paths = [train_image_dir + s for s in trainFilenames]

# 初始化变量，导入图片
x_train = []
train_labels = []
im_size = 256
img_height=im_size
img_width=im_size
bs = 32
i = 0
for f, breed in tqdm(y_train.values):
    img = load_img(train_image_dir + '{}.jpg'.format(f), target_size=(img_height, img_width))
    # 要把img转成向量然后才能append到x_train
    img = np.array(img, dtype="int64")
    x_train.append(img)
    label = one_hot_labels[i]
    train_labels.append(label)
    i += 1

#调整格式
y_train = np.array(train_labels, np.uint8)
x_train = np.array(x_train, np.float32) / 255.0

#模型
input_tensor = Input(shape=(256, 256, 3))
base_model = ResNet50(input_tensor=input_tensor,include_top=False,weights='imagenet')
#base_model = ResNet50(input_tensor=input_tensor,include_top=False,weights=None)
get_resnet50_output = K.function([base_model.layers[0].input, K.learning_phase()],
                          [base_model.layers[-1].output])
resnet50_train_output = []
delta = 10
for i in range(0,len(x_train),delta):
    print (i)
    one_resnet50_train_output = get_resnet50_output([x_train[i:i+delta], 0])[0]
    resnet50_train_output.append(one_resnet50_train_output)
resnet50_train_output = np.concatenate(resnet50_train_output,axis=0)
f = h5py.File('resnet50_train_output','w')
f.create_dataset('x_train', data = resnet50_train_output)
f.create_dataset('y_train', data = y_train)
f.close()

