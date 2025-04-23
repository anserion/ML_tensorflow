# --------------------------------------------------
# Базовые математические операции в TensorFlow
# --------------------------------------------------
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Создание числовых констант
a = tf.constant(2)
b = tf.constant(3)

addition = a + b  # Сложение
subtraction = a - b  # Вычитание
multiplication = a * b  # Умножение
division = a / b  # Деление
exponentiation = a ** b  # Возведение в степень

greater_than = a > b  # Больше
less_than = a < b  # Меньше
equal_to = tf.equal(a,b) # Равно
not_equal_to = tf.not_equal(a,b)  # Не равно

# создание матриц
matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])

# Сложение и умножение матриц
matrix_addition = matrix_a + matrix_b
matrix_multiplication = tf.matmul(matrix_a, matrix_b)

with tf.compat.v1.Session() as sess:
    print('a=', sess.run(a))
    print('b=', sess.run(b))

    result = sess.run(addition)
    print('addition:', result)

    print('subtraction:', sess.run(subtraction))
    print('multiplication:', sess.run(multiplication))
    print('division:', sess.run(division))
    print('exponentiation:', sess.run(exponentiation))

    print('greater_than:', sess.run(greater_than))
    print('less_than:', sess.run(less_than))
    print('equal_to:', sess.run(equal_to))
    print('not_equal_to:', sess.run(not_equal_to))

    print()

    print('matrix_a=', sess.run(matrix_a))
    print('matrix_b=', sess.run(matrix_b))
    print('matrix addition:', sess.run(matrix_addition))
    print('matrix mult:', sess.run(matrix_multiplication))
