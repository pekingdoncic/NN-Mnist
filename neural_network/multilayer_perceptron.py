import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient
from abc import ABC,abstractmethod

class MultilayerPerceptron:#多层感知机
    def __init__(self,data,labels,layers,normalize_data =False):#初始化
        #标签传入：数据，标签，层次（多层），规范化编码
        #预处理数据传入
        data_processed = prepare_for_training(data,normalize_data = normalize_data)[0]
        #然后赋值数据数据
        self.data= data_processed
        self.labels= labels #标签，因为是一个分类的类型，就需要one hot编码！
        self.layers= layers #784=28*28*1（输入数据） 25个隐藏神经元个数！784个特征转化为25维向量（自定义） 10（最后分类成10个！）
        self.normalize_data= normalize_data  #初始化赋值操作
        self.thetas = MultilayerPerceptron.thetas_init(layers) #初始化权重参数
        #多组权重操作

    def thetas_init(layers):#初始化权重的方法,就是wx+b中的那个w！
        num_layers=len(layers) #乘的个数
        thetas={}#多少层，有多少组对应的权重参数
        for layer_index in range(num_layers-1):#三层，就两组权重参数
            #前一层和后一层的值
            in_count=layers[layer_index]
            out_count=layers[layer_index+1]#
            # 这里需要考虑到偏置项，记住一点偏置的个数跟输出的结果是一致的，所以会in_count+1
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05  # 随机进行初始化操作，值尽里小一点
            #初始化的参数值希望值越小越好！
            #两组：25*785 10*26
        return thetas#返回初始化的参数（字典）


    def predict(self, data):
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]#数据预处理
        num_examples = data_processed.shape[0]# 获取数据集中的样本数量
        # 使用多层感知器（Multilayer Perceptron）进行前向传播
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed, self.thetas, self.layers)
       # print(type(np.argmax(predictions, axis=1).reshape((num_examples, 1))))
        # 找到每个样本的预测类别（以索引形式表示），并将结果重新形状为列向量
        # 这是通过找到具有最高值的神经元的索引来实现的
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iterations=1000, alpha=0.1):#训练模块，alpha，学习率
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)#对参数进行变换，从矩阵变成向量
        (optimized_theta,cost_history) = MultilayerPerceptron.gradient_decent(self.data,self.labels,unrolled_theta,self.layers,max_iterations,alpha)
        #反向传播
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history

    @staticmethod
    def thetas_unroll(thetas):#把矩阵转化成向量，矩阵拉成条，因为权重参数是一个矩阵，后续的代码级的操作不方便，所以要换成高维向量
        num_theta_layers = len(thetas)#多少个thetas！多少组
        unrolled_theta = np.array([])#为了循环一致，创建一个空的数组，为了方便拼接
        for theta_layer_index in range(num_theta_layers):
            #这里，利用的hstack将向量拼接在一起，利用的flatten将矩阵展平成为高纬向量
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))  # hstackQ.没行堆叠。
        #hstack：拼接在一起！
        return unrolled_theta



    @staticmethod#一个梯度下降完成的结果
    def gradient_decent(data,labels,unrolled_theta,layers,max_iterations,alpha):
        optimized_theta=unrolled_theta#最终所需要的结果
        cost_history=[]#每一次迭代过程中记录
        for _ in range(max_iterations):#迭代多少次
            #一长条向量转化成矩阵了
            #计算损失函数
            cost = MultilayerPerceptron.cost_function(data,labels,MultilayerPerceptron.thetas_roll(optimized_theta, layers), layers)
            cost_history.append(cost)#将计算的结果保存在cost_history结尾
            #计算更新：逐层运算
            theta_gradient = MultilayerPerceptron.gradient_step(data,labels,optimized_theta,layers)
            optimized_theta = optimized_theta - alpha* theta_gradient
        return optimized_theta,cost_history#返回更新后的参数以及损失值

    @staticmethod
    def cost_function(data,labels,thetas,layers):
        num_layers = len(layers)#计算多少层
        num_examples = data.shape[0]#样本个数
        num_labels = layers[-1]#输出层个数

        # 前向传播走一次
        predictions = MultilayerPerceptron.feedforward_propagation(data,thetas,layers)
      #  print(predictions)
        # 制作标签，每一个样本的标签都得是one-hot,向量
        bitwise_labels = np.zeros((num_examples,num_labels))#整理好维度
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
            #制作标签，对应的位置为1！
            #正确的位置，更接近1，错误的位置，更接近0！
        # 判断错误类别的损失值
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        #判断正确类别的损失值
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
       # print(bit_set_cost)
       # print(bit_not_set_cost)
        #计算损失值，对损失值标准化，取平均
        cost = (-1 / num_examples) * (bit_set_cost+bit_not_set_cost)
      #  print(cost)
        return cost

    @staticmethod
    def feedforward_propagation(data,thetas,layers):
        # 获取神经网络的总层数（num_layers）和输入数据的样本个数（num_examples）。
        num_layers=len(layers)
        num_examples=data.shape[0]
        #2.	初始化输入层的激活值为输入数据 in_layer_activation。
        in_layer_activation=data

        #逐层计算
        for layer_index in range(num_layers-1):
            theta=thetas[layer_index]#权重参数得到
            out_layer_activation=sigmoid(np.dot(in_layer_activation,theta.T))#矩阵计算，dot是矩阵相乘的意思！

            #得到第一个隐藏的结果
            #正常计算完之后num_examples*25，但是要考虑偏置项,变成num_examples*26
            out_layer_activation=np.hstack((np.ones((num_examples,1)),out_layer_activation))
            in_layer_activation=out_layer_activation

        #返回输出层结果,结果中不要偏执项
        return in_layer_activation[:,1:]


    @staticmethod
    #就是把高维向量转化成为原来的矩阵，方便与x进行矩阵直接的乘法！
    def thetas_roll(unrolled_thetas,layers):
        num_layers=len(layers)
        thetas={}
        unrolled_shift=0#矩阵变换之后现在哪里出了，标志符，把两组矩阵分开
        for layer_index in range(num_layers-1):
            in_count=layers[layer_index]
            out_count= layers[layer_index+1]
            #矩阵的长宽
            thetas_width=in_count+1 #784+1
            thetas_height=out_count #25
            #矩阵占用空间大小
            thetas_volume=thetas_height*thetas_width
            start_index=unrolled_shift
            end_index=unrolled_shift+thetas_volume
            #取出来一个矩阵
            layer_theta_unrolled=unrolled_thetas[start_index:end_index]
            thetas[layer_index]=layer_theta_unrolled.reshape((thetas_height,thetas_width))
            #更新这个值
            unrolled_shift=unrolled_shift+thetas_volume
        return thetas
    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)#还原成矩阵
        #反向传播，更新数据！
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data, labels, theta, layers)

        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)#向量还原成矩阵!
        return thetas_unrolled_gradients

    def back_propagation(data,labels,thetas,layers):
        num_layers=len(layers)
        #样本个数和特征数据，1700 785
        (num_examples,num_features)=data.shape
        num_labels_types=layers[-1]#最后一层
        deltas={}
        # 初始化操作：为每一层的权重微调矩阵初始化参数
        for layer_index in range(num_layers-1):
            in_count=layers[layer_index]
            out_count=layers[layer_index+1]
            deltas[layer_index]=np.zeros((out_count,in_count+1))#初始化参数
            #25 785 10*26
        for examples_index in range(num_examples):#对每个样本遍历，除了输出层。
            layers_inputs={}#创建两个空字典，用于存储每层的输入和激活值。
            layers_activations={}#得到的结果
            layers_activation=data[examples_index,:].reshape((num_features,1))#785*1
            #获取当前样本的输入数据，并进行形状调整，将其视为一列。
            layers_activations[0]=layers_activation#第0层，赋值
            #将输入数据保存为第0层的激活值。
            #逐层计算
            for layers_index in range(num_layers-1):#遍历每个训练样本
                layers_theta=thetas[layers_index]#得到当前权重参数值 25*785 10*26
                #第一层的输出，下一层的输入
                layer_input=np.dot(layers_theta,layers_activation)#第一次25*1 第二次10*1
                #完成激活函数，并且保存
                layers_activation=np.vstack((np.array([[1]]),sigmoid(layer_input)))#785*1
                #用激活函数（这里使用了Sigmoid函数），并添加偏置单元（1）到激活值中。
                layers_inputs[layers_index+1]=layer_input#后一层计算结果
                layers_activations[layers_index+1]=layers_activation#后一层经过激活函数的结果

            #最后一次得到的结果，取结果的时候，不要偏置参数
            output_layer_activation=layers_activation[1:,:]

            delta={}
            #标签处理
            bitwise_label = np.zeros((num_labels_types,1))#初始化
            bitwise_label[labels[examples_index][0]]= 1#对当前的样本赋值
            #根据当前样本的标签，将对应位置的标签值设置为1，以表示正确类别。

            #计算输出层和真实值之间的差值
            delta[num_layers-1]=output_layer_activation-bitwise_label
            # 遍历循环 L L-1 L-2
            #逆序遍历每一层，从倒数第二层到第1层。
            for layer_index in range(num_layers - 2, 0, -1):#遍历两次！
                layer_theta = thetas[layer_index]#获取当前层的权重矩阵。
                next_delta = delta[layer_index + 1]#获取下一层的误差信号。
                layer_input = layers_inputs[layer_index]#获取当前层的输入。
                layer_input = np.vstack((np.array((1)), layer_input))#在输入中添加偏置参数。
                 # 按照公式进行计算
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                #计算当前层的误差信号，应用了反向传播算法中的公式。
                # 过滤掉偏置参数
                delta[layer_index] = delta[layer_index][1:, :]
            for layer_index in range(num_layers  - 1):#更新参数，计算每层的权重微调
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)
                #上面是微调部分的参数，得到矩阵
                deltas[layer_index] = deltas[layer_index] + layer_delta  # 第-次25*785 第二次10*26
                #最终经过微调得到的结果。
        for layer_index in range(num_layers - 1): #对每层的权重微调矩阵进行平均，以获得最终的微调结果
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)
        return deltas
