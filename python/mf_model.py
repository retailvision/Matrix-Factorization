# код разработан на основе статьи
# https://www.r-craft.org/r-news/simple-matrix-factorization-with-tensorflow/

import numpy as np
import tensorflow as tf
import pandas as pd
import os

import sys

path = os.path.dirname(os.path.abspath(__file__))

# читаем данные из файла rating.csv
df = pd.read_csv(path + '/rating.csv', sep='\t', names=['customer', 'item', 'rate', 'time'])

num_customers = len(pd.Series(df.customer).unique()) # количество уникальных покупателей
num_items = len(pd.Series(df.item).unique()) # количество уникальных товаров

customer_indecies = [x-1 for x in df.customer.values] # номера покупателей идут с 1, надо превратить их в индексы с 0
item_indecies = [x-1 for x in df.item.values] # товары аналогично покупателям
rates = df.rate.values  # вектор оценок

# размер вектора коэффициентов
feature_len = 50

# каждому покупателю сопоставляем вектор коэффициентов размером feature_len (строим матрицу "П")
C = tf.Variable(initial_value=tf.truncated_normal([num_customers, feature_len]), name='customers')

# каждому товару сопоставляем вектор коэффициентов размером feature_len  (строим транспонированную матрицу "Т")
I = tf.Variable(initial_value=tf.truncated_normal([feature_len, num_items]), name='items')

# матрица вероятных оценок "О", получаемая путем перемножения П*Т
result = tf.matmul(C, I) 

# вытяним матрицу оценок в одномерный вектор, т.е. возьмем первую строку, присоединим к ней вторую и так до конца
# получится ряд чисел длиною num_customers*num_items
result_flatten = tf.reshape(result, [-1])

# т.к. мы вытянули матрицу оценок в одномерный вектор, то для того, чтобы узнать какую оценку поставил покупатель товару,
# надо индекс покупателя (номер строки матрицы, начиная с 0) умножить на кол-во товаров
# так мы перескочим на место, в горизонтальном векторе, откуда начинаются оценки данного покупателя
# затем, необходимо добавить индекс товара и окажемся на ячейке оценки искомого товара конкретным покупателем
R = tf.gather(result_flatten, customer_indecies * tf.shape(result)[1] + item_indecies, name='extracting_customer_rate')


# РАСЧЕТ ОШИБКИ


# вектор разницы между реальными оценками rates и расчитываемыми R
diff_op = tf.subtract(R, rates, name='trainig_diff')

# модуль вектора ошибок
diff_op_squared = tf.abs(diff_op, name="squared_difference")

# суммируем модуль вектора ошибок в одну итоговую сумму
# это будет итоговая ошибка предсказания, к уменьшению которой надо стремиться
cost = tf.reduce_sum(diff_op_squared, name="sum_squared_error")


# ОБУЧЕНИЕ


# скорость обучения на начальном этапе
lr = tf.constant(.001, name='learning_rate')

# скорость обучения будет меняться каждый раз по прохождению 10000 шагов 
# см. exponential_decay (адаптивная скорость обучения)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)

# будем минимизировать суммарную ошибку (cost) по алгоритму градиентного спуска
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(cost, global_step=global_step)

# инициализируем tf-сессию
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# запуск обучения на 1000 эпох
for i in range(1000):
    sess.run(training_step)


# ПРИМЕР


# лучшие 5 оценок покупателя с кодом 1
customer_code = 1 # код
customer_index = customer_code - 1 # код в индекс
known_rates = df[df['customer'] == customer_code].item.values # товары, которые покупатель покупал в реальности
items_ids = [x-1 for x in pd.Series(df.item[~df.item.isin(known_rates)]).unique()] # вектор индексов товаров, которые покупатель еще ни разу не покупал
customer_by_code_rates = tf.gather(result_flatten, tf.add(items_ids, customer_index * num_items), name='customer_by_code_rates') # выбираем оценки покупателя по неизвестным товарам

np_rates = sess.run(customer_by_code_rates) # старт tf-сессии

# результат в массив numpy типа key-value
dtype = [("index", int), ("value", float)]
np_rates_table = np.empty(len(items_ids), dtype)
np_rates_table['index'] = np.arange(len(items_ids))
np_rates_table['value'] = np_rates

# сортируем по значению (оценке)
np_rates_table = np.sort(np_rates_table, order=['value'], axis=0)

# выводим индексы последних (сымых высоких) оценок
print(np_rates_table[-5:]['index'])