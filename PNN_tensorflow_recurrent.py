# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# рекуррентного типа в библиотеке TensorFlow (не работает)
# --------------------------------------------------
# AttributeError: `BasicLSTMCell` is not available with Keras 3.
# AttributeError: `LSTMCell` is not available with Keras 3.
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

n_samples=100
# ------------------------
# Создание графа нейросети
# ------------------------
# входы нейросети
x=tf.compat.v1.placeholder(tf.float32, [n_samples,])
y=tf.compat.v1.placeholder(tf.float32, [1,])

# две рекуррентные ячейки LSTM
rnn_cell1=tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=64)
rnn_cell2=tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=64)

# связывание ячеек LSTM в нейросеть
rnn_layer = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells=[rnn_cell1,rnn_cell2])

# встраивание рекуррентной нейросети в общий граф
outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell=rnn_layer,
                                                   inputs=x,
                                                   dtype=tf.float32)

# выходной полносвязный слой (1 нейрон, 10 входов)
tmp=final_state.get_shape().as_list()
mlp_inputs=tmp[-1]
mlp_input=final_state[-1][-1]
W_output=tf.Variable(tf.random.normal((mlp_inputs,1)))
B_output=tf.Variable(tf.random.normal((1,)))
# внутренняя структура выходного слоя (функция активации sigmoid)
y_output=tf.nn.sigmoid(tf.matmul(mlp_input,W_output)+B_output)
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
    # Генерация входных данных для обучения
    x_train = np.random.rand(n_samples)
    y_train = np.random.randint(0,2,n_samples)

    # собственно процесс обучения
    n_epochs=10
    for i in range(n_epochs):
        _,loss=sess.run([train_op,f_loss],{x:x_train,y:y_train})
        print('train loss value:',loss)
    # --------------------

