import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import sklearn
import matplotlib.pyplot as plt

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Activation
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.optimizers import SGD, Adam

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
bs = 64
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
y_train_raw = np.array(train_labels, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.0
# 分割训练和验证数据集
X_train, X_val, Y_train, Y_val = train_test_split(x_train_raw, y_train_raw, test_size=0.1, random_state=0)
# 下面这是被我抛弃的那个五十多层的网络....
#classifier = ResNet50(include_top=False, weights='imagenet',
#                   input_shape=(img_height,img_width, 3))
#增加一层，将ResNet50的输出作为输入
#x = Flatten(name='flatten')(classifier.output)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
#x = Dense(200, activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
#x = Dense(120, activation='softmax')(x)
#my_model = Model(inputs=classifier.input, outputs=x)
#my_model.summary()

#构建模型
model = Sequential()
#第一层为二维卷积层
# 32 为filters卷积核的数目，也为输出的维度
# kernel_size 卷积核的大小，3x3
# 激活函数选为relu
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_height,img_width, 3)))
# 再加一层卷积，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 加Dropout，断开神经元比例为25%
model.add(Dropout(0.25))
# 加Flatten，数据一维化
model.add(Flatten())
# 加Dense，输出128维
model.add(Dense(16, activation='relu'))
# 再一次Dropout
model.add(Dropout(0.5))
# 最后一层为Softmax
model.add(Dense(120, activation='softmax'))

#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer=keras.optimizers.Adadelta(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs=5,batch_size=bs, validation_data=(X_val, Y_val), verbose=1)
