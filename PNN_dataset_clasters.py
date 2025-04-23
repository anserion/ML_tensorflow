# --------------------------------------------------
# создание синтетического набора данных для обучения
# многослойного персептрона (два разделимых кластера)
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

n_samples = 1000
# ------------------------------
# Генерация данных для обучения
# ------------------------------
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

# ------------------------------
# Построение визуализации
# ------------------------------
xx=[p[0] for p in x_train ]
yy=[p[1] for p in x_train ]
plt.scatter(xx,yy, color='blue', label='Train')

plt.legend()
plt.show()
