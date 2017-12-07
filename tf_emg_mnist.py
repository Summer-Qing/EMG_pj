# coding utf-8

import tensorflow as tf

NUM = 5  # 定义需要考察的动作数量

# 输入数据是一个 不限个数（行数），每个数据有八列;
# None 表示可以输入任意个数据

x = tf.placeholder(tf.float32, [None, 8])

# 开始预测五个动作
Weight = tf.Variable(tf.zeros([8, NUM]))
bias = tf.Variable(tf.zeros([NUM]))

# 预测值
y = tf.nn.softmax(tf.matmul(x, Weight) + bias)

# #################### 模型定义完毕 ####################

# #################### 损失函数 ####################
y_ = tf.placeholder("float", [None, NUM])  # 预测NUM个动作
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# #################### 反响传播函数 ####################
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# #################### 开始执行 ####################
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# #################### 抓取数据 重复训练 ##############
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
