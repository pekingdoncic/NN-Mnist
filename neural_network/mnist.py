import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math

from multilayer_perceptron import MultilayerPerceptron

data = pd.read_csv('../data/mnist-demo.csv')
# data1 = pd.read_csv('../data/iris.csv')
# print(data)
# # print(data1)
# print(type(data))
numbers_to_display = 25#展示用 5行5列
num_cells = math.ceil(math.sqrt(numbers_to_display))#
plt.figure(figsize=(10,10))#大表中，很多子图


#展示图片存储的数据，
for plot_index in range(numbers_to_display):#展示很多图像
    digit = data[plot_index:plot_index+1].values#拿到像素值数据
    digit_label = digit[0][0]#在csv文件中位于第一列
    digit_pixels = digit[0][1:]#像素点的值
    image_size = int(math.sqrt(digit_pixels.shape[0]))#图像大小
    frame = digit_pixels.reshape((image_size,image_size))#展示图形参数
    plt.subplot(num_cells,num_cells,plot_index+1) #长宽和当前的位置
    plt.imshow(frame,cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.5,hspace=0.5)#调整间距
plt.show()


train_data = data.sample(frac = 0.8)#frac为比例
test_data = data.drop(train_data.index)#测试集为剩下的值

train_data = train_data.values
test_data = test_data.values

num_training_examples = 500#训练使用的例子
#定义测试集和训练集
x_train = train_data[:num_training_examples,1:]#从第二列开始
y_train = train_data[:num_training_examples,[0]]#只取标签

x_test = test_data[:,1:]
y_test = test_data[:,[0]]

#自定义神经网络的层次和每一层的神经元个数
layers = [784,50,10]
#定义对数据进行标准化处理，迭代次数，学习率
normalize_data = True
max_iterations = 1500#可以小一些，训练比较快
alpha = 0.1#初始值

#传输参数，开始训练模型
multilayer_perceptron = MultilayerPerceptron(x_train,y_train,layers,normalize_data)#初始化赋值
(thetas,costs) = multilayer_perceptron.train(max_iterations,alpha)
plt.plot(range(len(costs)),costs)
plt.xlabel('Grident steps')#梯度下降
plt.ylabel('costs')#损失值
plt.show()

#获得预测值
y_train_predictions = multilayer_perceptron.predict(x_train)
y_test_predictions = multilayer_perceptron.predict(x_test)

#计算准确率：正确预测的样本数(利用布尔函数提取正确预测的样本)除以总样本数，并乘以 100，计算出训练集和测试集的准确率（以百分比表示）。
train_p = np.sum(y_train_predictions  ==y_train)/y_train.shape[0]*100
test_p = np.sum(y_test_predictions  ==y_test)/y_test.shape[0]*100
print('训练集准确率',train_p)
print('测试集准确率',test_p)

#遍历每一张要展示的图片，将预测正确与否展现出来。
numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15,15))
for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index,0]
    digit_pixels = x_test[plot_index,:]

    predicted_label = y_test_predictions[plot_index][0]

    image_size = int(math.sqrt(digit_pixels.shape[0]))

    frame = digit_pixels.reshape((image_size,image_size))

    color_map = 'Greens' if predicted_label == digit_label else 'Reds'
    plt.subplot(num_cells,num_cells,plot_index+1)
    plt.imshow(frame,cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.show()