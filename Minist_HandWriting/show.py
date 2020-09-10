from tensorflow.examples.tutorials.mnist import  input_data
import pylab
import tensorflow as tf
from datetime import datetime

startTime = datetime.now()

mnist = input_data.read_data_sets("MINST_daya/", one_hot=True)

tf.reset_default_graph()

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # mnist data 维度28*28=784
y = tf.placeholder(tf.float32, [None, 10])

# 设置模型的权重
W = tf.Variable(tf.random_normal([784, 10]))  # W的维度是[784, 10]
b = tf.Variable(tf.zeros([10]))
# 定义输出节点， 构建模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()

model_path = "mnist/521model.ckpt"

# 读取模型
with tf.Session() as sess3:
    # 初始化参数
    sess3.run(tf.global_variables_initializer())
    #从保存的模型中获取权重
    saver.restore(sess3, model_path)
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("准确度：",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
    output = tf.argmax(pred, 1)

    # 其中的3为得到几个值
    num = 3

    batch_xs, batch_ys = mnist.train.next_batch(num)
    outputval, predv = sess3.run([output, pred], feed_dict={x:batch_xs})
    print(outputval, pred, batch_ys)

    i = 0

    while (i <= num):           
        im = batch_xs[i]
        im = im.reshape(-1, 28)
        pylab.imshow(im)
        pylab.show()
        i += 1

print("Time taken:", datetime.now() - startTime)