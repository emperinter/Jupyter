import sys
import numpy as np
import tensorflow as tf
from datetime import datetime


# 在代码中加入如下代码，忽略警告：
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
# This works equally with tf.Session() instead of tf.InteractiveSession() if you prefer.


# log_device_placement=Ture
# 会告诉我们当前是什么来运算的

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
with tf.compat.v1.Session() as session:
        result = session.run(sum_operation)
        print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)