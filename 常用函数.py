import tensorflow as tf
import numpy as np
a = tf.constant([[1, 5, 4], [1, 4, 2]])
print(a)
print(a.dtype)       # shape=(2, )  逗号前隔开了几个数字就是几维的，此逗号隔开了一个数字，就是一维的，2表示有两个数值
print(a.shape)


# 将numpy的数据类型转换为tensor格式
b = np.arange(0, 5)
c = tf.convert_to_tensor(b, dtype=tf.int64)
print(b)
print(c)

# 创建一个tensor
'''
创建全为0的张量  tf.zeros(维度)
创建全为1的张量  tf.ones(维度)
创建全为指定值的张量 tf.fill(维度，指定数)
维度：一维 直接写个数，二维 用[行，列]，多维 用[n, m, j, ...]方括号里写每个维度的个数
生成正态分布的随机数，默认均值为0，标准差为1  tf.random.normal(维度，mean=均值，stddev=标准差)
生成截断式正态分布的随机数 tf.random.truncated_normal(维度，mean=均值，stddev=标准差)
生成指定维度的均匀分布随机数 tf.random.uniform(维度，minval=最小值，maxval=最大值)前闭后开
'''
a0 = tf.zeros([2, 3])
a1 = tf.ones(4)
a2 = tf.fill([2, 2], 9)

a3 = tf.random.normal([2, 2], mean=0.5, stddev=1)
a4 = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)  # 生成数据均在两倍标准差内，更像0.5集中
a5 = tf.random.uniform([2, 2], minval=0, maxval=1)

'''
tf.cast(张量名, dtype = 数据类型) 强制tensor转换为该数据类型
tf.reduce_min(张量名) 计算张量维度上元素的最小值
tf.reduce_max(张量名) 计算张量维度上元素的最大值 
'''

a6 = tf.constant([1., 2., 3.], dtype=tf.float64)
print(a6)
a7 = tf.cast(a6, tf.int32)
print(a7)
print(tf.reduce_max(a7))

'''
axis控制执行维度
axis=0表示按纵向进行操作，axis=1表示按横向进行操作
'''

a8 = tf.constant([[1, 2, 3],
                  [2, 2, 3]])
print(a8)
print(tf.reduce_mean(a8))
print(tf.reduce_mean(a8, axis=0))

'''
tf.Variable()将变量标记为’可训练‘，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练参数
'''
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))

'''
常用的数学运算
+ - * / = tf.add tf.subtract tf.multiply tf.divide   只有维度相同的张量才可以做四则运算
平方，次方，开方= tf.square, tf.pow, tf.sqrt
矩阵乘=tf.matmul
'''

x = tf.ones([1, 3])
y = tf.fill([1, 3], 3.)
print(x)
print(y)
print(tf.add(x, y))
print(tf.subtract(x, y))
print(tf.multiply(x, y))
print(tf.divide(x, y))

z = tf.fill([1, 2], 3.)
print(z)
print(tf.square(z))
print(tf.pow(z, 3))
print(tf.sqrt(z))

q = tf.fill([3, 2], 2.)
p = tf.fill([2, 3], 3.)
print(q)
print(p)
print(tf.matmul(q, p))

'''
tf.data.Dataset.from_tensor_slices((输入特征，标签))切分传入张量的第一维度，生成输入特征/标签对，构建数据集
适用于tensor和numpy格式
'''

features = tf.constant([10, 23, 56, 45])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)


