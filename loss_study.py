import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))    # 保存和更新神经网络中的参数使用tf.Variable（）函数，tf.constant(张量内容, dtype=数据类型)(创建一个常张量，传入list或数值)
lr = 0.2
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 w -= lr*grads
    print('After %s epoch,w is %f,loss is %f' % (epoch, w.numpy(), loss))