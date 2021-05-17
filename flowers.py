import tensorflow as tf
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

# 导入数据，分别输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据 seed：随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
# print(x_train)
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，转换为tensor
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 将特征和标签一一对应(把数据集分批次，每次batch组数据)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络参数，4个输入特征，输入层为4个输入节点；分三类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1
train_loss_results = []   # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在列表中，画acc曲线
epoch = 500
loss_all = 0  # 每轮分4个step，记录四个step生成的4个loss的和

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):    #每一个step一个batch
        with tf.GradientTape() as tape:    # with记录梯度信息
            y = tf.matmul(x_train, w1) + b1    # 神经网络乘加计算
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)   # 将标签值转换为独热码格式，方便计算loss和acc
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()    # 将每个step计算出的loss累加
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # 每个epoch，打印loss信息
    print('Epoch {}, loss: {}'.format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0

    # 测试部分
    # total_correct为预测对的样本个数，total_number为测试的总样本数
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)   # 返回y中最大值的索引，即预测分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1,否则为0，将bool类型的结果转换为int
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)   # .equal判断是否相等
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch的correct加起来
        total_correct += int(correct)
        total_number = x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print('Test_acc:', acc)
    print('--------------------------')

# 绘制loss曲线
plt.title('loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()
plt.show()

# 绘制acc
plt.title('Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Acc$')
plt.legend()
plt.show()
