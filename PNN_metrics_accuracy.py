# --------------------------------------------------
# оценка точности (обучения) нейронной сети
# с визуализацией процесса увеличения точности3
# --------------------------------------------------
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Создание графа нейросети
# ------------------------
# входы нейросети
n_inputs=2 # количество входов (координаты точки)
x=tf.compat.v1.placeholder(tf.float32,(None,n_inputs))
# выходы нейросети
y=tf.compat.v1.placeholder(tf.float32,(None,))

# слой 1
n_hidden1=5 # число нейронов в слое1
W1=tf.Variable(tf.random.normal((n_inputs,n_hidden1)))
B1=tf.Variable(tf.random.normal((n_hidden1,)))
# внутренняя структура слоя 1 (функция активации ReLU)
y1=tf.nn.relu(tf.matmul(x,W1)+B1)

# слой 2
n_hidden2=5
W2=tf.Variable(tf.random.normal((n_hidden1,n_hidden2)))
B2=tf.Variable(tf.random.normal((n_hidden2,)))
# внутренняя структура слоя 2 (функция активации ReLU)
y2=tf.nn.relu(tf.matmul(y1,W2)+B2)

# выходной слой (1 нейрон)
W_output=tf.Variable(tf.random.normal((n_hidden2,1)))
B_output=tf.Variable(tf.random.normal((1,)))
# внутренняя структура выходного слоя (функция активации sigmoid)
y_output=tf.nn.sigmoid(tf.matmul(y2,W_output)+B_output)
y_predict=tf.round(y_output)

# конструируем функцию потерь, которую нужно минимизировать, обучая нейросеть
y_expand=tf.compat.v1.expand_dims(y,1)
entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=y_expand,
                                                logits=y_output)
f_loss=tf.reduce_sum(entropy)

# задаем алгоритм минимизации (настройки W и B)
train_op=tf.compat.v1.train.AdamOptimizer(.001).minimize(f_loss)
# ------------------------------
# конец описания графа нейросети
# ------------------------------

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # --------------------
    # Обучение нейросети
    #--------------------
    n_samples = 1000
    # Генерация входных данных для обучения
    x_zeroes=np.random.multivariate_normal(mean=np.array((0.25,0.25)),
                                           cov=0.01*np.eye(2),
                                           size=n_samples//2)
    y_zeroes=np.zeros((n_samples//2,))
    x_ones = np.random.multivariate_normal(mean=np.array((0.75, 0.75)),
                                           cov=0.01 * np.eye(2),
                                           size=n_samples // 2)
    y_ones = np.ones((n_samples // 2,))
    x_train = np.vstack([x_zeroes,x_ones])
    y_train = np.concatenate([y_zeroes,y_ones])

    # собственно процесс обучения
    n_epochs=1000
    val_accuracy=[]
    for i in range(n_epochs):
        _,predict=sess.run([train_op,y_predict],
                        {x:x_train,y:y_train})
        true_predictions=0
        for k in range(n_samples):
            if predict[k]==y_train[k]:
                true_predictions=true_predictions+1
        accuracy=true_predictions/n_samples
        val_accuracy.append(accuracy)
    # --------------------

    # Построение визуализации точности при обучении классификатора
    xx=[i for i in range(n_epochs) ]
    plt.scatter(xx,val_accuracy, color='blue', label='Train accuracy')

    plt.legend()
    plt.show()
