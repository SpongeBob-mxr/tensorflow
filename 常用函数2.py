import tensorflow as tf
import numpy as np
'''
tf.GradientTape
with结构记录计算过程，gradient求出张量的梯度
with tf.GradientTape() as tape:
    若干个计算过程
grad = tape.gradient(函数，对谁求导)
'''

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)
grad = tape.gradient(loss, w)
print(grad)

'''
enumerate(列表名)，遍历每个元素(如：列表，元组，字符串)，组合为：索引 元素，常在for循环中使用
'''
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)

'''
tf.one_hot()函数将待转换数据，转换成one_hot形式的数据输出
tf.one_hot(待转换的数据， depth=几分类)
'''
classes = 3
labels = tf.constant([1, 2, 0])
output = tf.one_hot(labels, classes)
print(output)

'''
tf.nn.softmax()当n分类的n个输出通过softmax函数便符合概率分布
'''
x1 = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(x1)
print('After softmax,y_pro is:', y_pro)

'''
assign_sub  赋值操作，更新参数的值并返回
调用assign_sub前，先用tf.Variable定义变量w为可训练（可自跟新）
w.assign_sub(w要自减的内容)
'''

w1 = tf.Variable(4)
w1.assign_sub(2)
print(w1)

'''
tf.argmax返回张量沿指定维度最大值的索引
tf.argmax(张量名, axis=操作轴)
'''
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print(test)
print(tf.argmax(test, axis=0))
print(tf.argmax(test, axis=1))