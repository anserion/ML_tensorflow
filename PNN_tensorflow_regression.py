# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# из одиночного нейрона (регрессия) в библиотеке TensorFlow
# --------------------------------------------------
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt

nSamples=100

# Генерация входных данных для обучения
x_train=np.random.rand(nSamples,1)
# Генерация выходных данных для обучения
k_train=np.random.rand()
b_train=0
noise_scale=0.03
noise=np.random.normal(scale=noise_scale,size=(nSamples,1))
y_train=np.reshape(k_train*x_train+b_train+noise,(-1))

# Создание графа нейросети
# ------------------------
# входы нейрона
x=tf.compat.v1.placeholder(tf.float32,(None,1))
y=tf.compat.v1.placeholder(tf.float32,(None,))
# веса нейрона W и смещение B
W=tf.Variable(tf.random.normal((1,1)))
B=tf.Variable(tf.constant(0,dtype=tf.float32))
# внутренняя структура нейрона
y_pred=tf.matmul(x,W) # + B # не работает, если раскомментировать +B
# функция потери, которую нужно минимизировать, обучая нейрон
f_loss=tf.reduce_sum((y-y_pred)**2)
# задаем алгоритм минимизации (настройки W и B)
train_op=tf.compat.v1.train.AdamOptimizer(0.01).minimize(f_loss)
# --------------------------

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # Обучение нейросети
    #--------------------
    for i in range(1000):
        _,loss,ww,bb=sess.run([train_op,f_loss,W,B],{x:x_train,y:y_train})
        print(f'{i}:, loss={loss}, w={ww}, real_w={k_train}, b={bb}')
    # --------------------

    # Тестирование работы нейросети
    # -----------------------------
    # Генерация данных для теста
    nTests=100
    x_test = np.random.rand(nTests,1)
    # расчет выходных значений обученного нейрона
    y_test = sess.run(y_pred, {x:x_test})
    # -----------------------------

# Построение графика
plt.scatter(x_train, y_train, color='blue', label='Train')
plt.scatter(x_test, y_test, color='red', label='Predicted')
plt.legend()
plt.show()
