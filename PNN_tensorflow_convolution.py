# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# сверточного типа в библиотеке TensorFlow
# --------------------------------------------------
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

n_samples = 5
# ------------------------
# Создание графа нейросети
# ------------------------
# входы нейросети
img_width=32
img_height=32
x=tf.compat.v1.placeholder(tf.float32,(n_samples,img_width,img_height,1))
# выходы нейросети
y_n=10
y=tf.compat.v1.placeholder(tf.float32,(n_samples,y_n))

# ----------------------
# первый сверточный блок
# ----------------------
filter1_width=5
filter1_height=5
ch_in1=1
ch_out1=32
W_conv1=tf.Variable(
    tf.compat.v1.truncated_normal(
        shape=(filter1_width, filter1_height, ch_in1, ch_out1),
        stddev=0.1,
        dtype=tf.float32)
)
B_conv1=tf.Variable(tf.zeros(shape=[ch_out1],dtype=tf.float32))
# внутренняя структура блока
# формат input "NHWC" (по умолчанию) или "NCHW"- 4D-тензор
# NHWC=(изображений в X, высота, ширина, число каналов (например, RGB - 3))
# NCHW=(изображений в X, число каналов, высота, ширина изображения)
# формат filters=(filter_rows,filter_columns,in_channels,out_channels)
# strides=[shift by img, vertical shift, horizontal shift, shift by channels]
# padding - "SAME" -  обрамлять нулями входное изображение при фильтрации
#           "VALID" - не обрамлять
conv1=tf.nn.conv2d(input=x,
                   filters=W_conv1,
                   strides=[1,1,1,1],
                   padding='SAME' # zeroes around input image
                   )
relu1=tf.nn.relu(tf.nn.bias_add(conv1,B_conv1))
pool1=tf.nn.max_pool(relu1,
                     ksize=[1, 2, 2, 1],  # размер pool-ядра в NHWC
                     strides=[1, 2, 2, 1],# смещения pool-ядра в NHWC
                     padding='SAME'
                     )

# ----------------------
# второй сверточный блок
# ----------------------
f2_w=5
f2_h=5
ch_in2=ch_out1
ch_out2=64
W_conv2=tf.Variable(
    tf.compat.v1.truncated_normal((f2_h, f2_w, ch_in2, ch_out2), stddev=0.1)
)
B_conv2=tf.Variable(tf.constant(value=0.1, shape=[ch_out2]))
# внутренняя структура блока
conv2=tf.nn.conv2d(pool1, W_conv2, strides=1, padding='SAME')
relu2=tf.nn.relu(tf.nn.bias_add(conv2,B_conv2))
pool2=tf.nn.max_pool(relu2, ksize=2, strides=2, padding='SAME')

# ----------------------
# полносвязный слой
# ----------------------
mlp_outputs=y_n # число нейронов (выходов)

# приводим выход сверточного блока к одномерному виду
tmp=pool2.get_shape().as_list()
mlp_inputs=tmp[1]*tmp[2]*tmp[3]
mlp_input=tf.reshape(pool2,[tmp[0],mlp_inputs])

W_output=tf.Variable(tf.random.normal((mlp_inputs,mlp_outputs)))
B_output=tf.Variable(tf.random.normal((mlp_outputs,)))
# внутренняя структура выходного слоя (функция активации sigmoid)
mlp_out=tf.squeeze(tf.matmul(mlp_input, W_output)+B_output)
mlp_logit=tf.nn.sigmoid(mlp_out)
mlp_predict=tf.round(mlp_logit)

# функция потерь выходного слоя
entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=mlp_out)
f_loss=tf.reduce_sum(entropy)

# задаем алгоритм минимизации (настройки W и B)
train_op=tf.compat.v1.train.AdamOptimizer(.01).minimize(f_loss)
# ------------------------------
# конец описания графа нейросети
# ------------------------------

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # --------------------
    # Обучение нейросети
    #--------------------
    # Генерация входных данных для обучения
    x_train = np.random.rand(img_width * img_height * n_samples)
    x_train = np.reshape(x_train, (n_samples, img_height, img_width, 1))

    y_train = np.zeros((n_samples,y_n))
    for i in range(n_samples):
        y_train[i,np.random.randint(0,y_n)]=1

    # собственно процесс обучения
    n_epochs=100
    for i in range(n_epochs):
        _,loss=sess.run([train_op,f_loss],{x:x_train,y:y_train})
        print('Train loss value:',loss)
    # --------------------

    # -----------------------------
    # Тестирование работы нейросети
    # -----------------------------
    predicts = sess.run(mlp_predict, {x:x_train})
    # вывод значений выходов нейросети
    for i,predict in enumerate(predicts):
        print(f'predict: {predict}, real: {y_train[i]}')
    # -----------------------------