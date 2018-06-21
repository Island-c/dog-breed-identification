import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import pandas as pd
import keras
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
from keras import backend as K
from tqdm import tqdm
from keras.optimizers import SGD, Adam

from keras.preprocessing.image import load_img

#从数据集提取数据
f = h5py.File('dataset','r')
resnet50_train_output = f['x_train'][:]
y_train= f['y_train'][:]
f.close()




#模型
input_tensor = Input(shape=(1, 1, 2048))  
x = Flatten(name='flatten')(input_tensor)
x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
#x = Dense(512, activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(0.6)(x)
x = Dense(120, activation='softmax')(x)
my_model = Model(inputs=input_tensor, outputs=x)
my_model.summary()
my_model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])

#提取模型
my_model.load_weights(os.path.join('cnn_model_resnet50'+'.h5'))

history=my_model.fit(resnet50_train_output, y_train, epochs=100, batch_size=128, validation_split=0.1, verbose=2)

#history=my_model.fit(resnet50_train_output, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test), verbose=1)

print(history.history.keys())

fig = plt.figure()#新建一张图
plt.plot(history.history['acc'],label='training acc')
#plt.plot(history.history['val_acc'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
#plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('loss.png')

#保存模型
my_model.save_weights(os.path.join('cnn_model_resnet50.h5'))

#预测
test = pd.read_excel("test.xlsx")
testFilenames=pd.read_excel("test.xlsx", usecols=0)
test_image_dir = "./test/"
# 转成numpy矩阵，再转成列表，取出文件名，组合成路径。...
testFilenames=testFilenames.T.as_matrix()
testFilenames=testFilenames.tolist()
testFilenames = [y for x in testFilenames for y in x]
test_img_paths = [test_image_dir + s for s in testFilenames]
# 初始化变量，导入图片
x_test = []
im_size = 256
img_height=im_size
img_width=im_size
i = 0
for f in tqdm(test_img_paths):
    img = load_img(f, target_size=(img_height, img_width))
    # 要把img转成向量然后才能append到x_train
    img = np.array(img, dtype="int64")
    x_test.append(img)
    i += 1
x_test = np.array(x_test, np.float32) / 255.0
###
resnet50_test_output = []  
delta = 10  

for i in range(0,len(X_test),delta):  

    print i  

    one_resnet50_test_output = get_resnet50_output([X_test[i:i+delta], 0])[0]  

    resnet50_test_output.append(one_resnet50_test_output)  

resnet50_test_output = np.concatenate(resnet50_test_output,axis=0)  

f = h5py.File(file_name,'w')            

f.create_dataset('resnet50_test_output', data = resnet50_test_output)  

f.close()  










