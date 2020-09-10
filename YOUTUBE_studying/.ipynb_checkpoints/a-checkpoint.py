# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 2019/5/17 17:03
# @Author  : emperinter
# @Desc    :  minist数据集下载
# @File    : mnist_data_load.py
# @Software: PyCharm
##################
# 数据下载
#################
from tensorflow.examples.tutorials.mnist import  input_data
import pylab

mnist = input_data.read_data_sets("MINST_daya/", one_hot=True)
print("输入数据：",mnist.train.images)
print("数据的shape：",mnist.train.images.shape)

# 展示数据集中的一张图片
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

from tensorflow.examples.tutorials.mnist import  input_data
import pylab
import tensorflow as tf
from datetime import datetime
mnist = input_data.read_data_sets("MINST_daya/", one_hot=True)
tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # mnist data 维度28*28=784
y = tf.placeholder(tf.float32, [None, 10])

# 定义学习参数
# 设置模型的权重
W = tf.Variable(tf.random_normal([784, 10]))  # W的维度是[784, 10]
b = tf.Variable(tf.zeros([10]))
# 定义输出节点， 构建模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)
# 定义反向传播的结构，编译训练模型，得到合适的参数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 参数设置/学习率
learning_rate = 0.1

# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
startTime = datetime.now()
#
# original
#
# training_epochs = 25   # 将整个训练样本迭代25次
# batch_size = 100    # 在训练过程中每次随机抽取100条数据进行训练
# display_step = 1   # 迭代的步数
# saver = tf.train.Saver()
# model_path = "mnist/521model.ckpt"

# me
training_epochs = 1800
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = "mnist/521model.ckpt"

# 开始训练
with tf.Session()  as sess:
    # 初始化节点
    sess.run(tf.global_variables_initializer())

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部的数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行和优化节点的损失函数值
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs,
                                                          y: batch_ys})
            # 计算平均损失值
            avg_cost += c / total_batch

        # 显示训练中的详细信息
        if (epoch+1) % display_step ==0:
            print("Epoch:","%04d"%(epoch+1), "cost=",'{:.9f}'.format(avg_cost))

    print("训练成功！！")

    # 模型测试
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("准确度：",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    # 保存模型的权重
    save_path = saver.save(sess, model_path)
    print("模型文件在：%s"%save_path)

print("Time taken:", datetime.now() - startTime)

# 该模块是第一次训练模块后的精准读来弄的

startTime = datetime.now()

# 读取模型
print("检验第一次训练的情况")
with tf.Session() as sess2:
    # 初始化参数
    sess2.run(tf.global_variables_initializer())
    ############################
    #
    #从保存的模型中获取权重
    #
    ###########################

    saver.restore(sess2, model_path)

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("准确度：",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess2.run([output, pred], feed_dict={x:batch_xs})
    print(outputval, pred, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

print("Time taken:", datetime.now() - startTime)

#
# ###################
# # 测试是否能在原有的基础上来提高训练精度
# ##################
# training_epochs = 1000
# batch_size = 100
# display_step = 1
# saver = tf.train.Saver()
# model_path = "mnist/521model.ckpt"
#
# startTime = datetime.now()
#
# # 读取模型
# print("启动第二次session")
# # Creates a session with log_device_placement set to True.
# with tf.Session() as sess2:
#     # 初始化参数
#     sess2.run(tf.global_variables_initializer())
#     #从保存的模型中获取权重
#     saver.restore(sess2, model_path)
#
#      # 启动循环开始训练
#     for epoch in range(training_epochs):
#         avg_cost = 0.
#         total_batch = int(mnist.train.num_examples/batch_size)
#         # 遍历全部的数据集
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             # 运行和优化节点的损失函数值
#             _, c = sess2.run([optimizer, cost], feed_dict={x:batch_xs,
#                                                           y: batch_ys})
#             # 计算平均损失值
#             avg_cost += c / total_batch
#
#         # 显示训练中的详细信息
#         if (epoch+1) % display_step ==0:
#             print("Epoch:","%04d"%(epoch+1), "cost=",'{:.9f}'.format(avg_cost))
#
#     print("训练成功！！")
#
#     # 模型测试
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # 计算准确率
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print("准确度：",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
#
#     # 保存模型的权重
#     save_path = saver.save(sess2, model_path)
#     print("模型文件在：%s"%save_path)
#
# print("Time taken:", datetime.now() - startTime)
#
# startTime = datetime.now()
#
# # 读取模型
# print("测试第二次训练的情况")
# with tf.Session() as sess3:
#     # 初始化参数
#     sess3.run(tf.global_variables_initializer())
#     #从保存的模型中获取权重
#     saver.restore(sess3, model_path)
#
#     # 测试 model
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # 计算准确率
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print("准确度：",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
#
#     output = tf.argmax(pred, 1)
#     batch_xs, batch_ys = mnist.train.next_batch(2)
#     outputval, predv = sess3.run([output, pred], feed_dict={x:batch_xs})
#     print(outputval, pred, batch_ys)
#
#     im = batch_xs[0]
#     im = im.reshape(-1, 28)
#     pylab.imshow(im)
#     pylab.show()
#
#     im = batch_xs[1]
#     im = im.reshape(-1, 28)
#     pylab.imshow(im)
#     pylab.show()
#
# print("Time taken:", datetime.now() - startTime)

