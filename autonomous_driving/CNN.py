#TensorFlow에서 제공하는 데이터로 MNIST, CIFAR10 연습
import tensorflow as tf

#MNIST
(mnist_x,mnist_y), _= tf.keras.datasets.mnist.load_data()
print(mnist_x.shape,mnist_y.shape)
# (60000, 28, 28) (60000,)

#CIFAR10
(cifar_x,cifar_y), _= tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape,cifar_y.shape)
# (50000, 32, 32, 3) (50000, 1)

