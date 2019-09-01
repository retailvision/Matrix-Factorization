# https://www.r-craft.org/r-news/simple-matrix-factorization-with-tensorflow/

import numpy
import tensorflow as tf
import pandas as pd
import os

import sys


# read data
df = pd.read_csv('~/tf/rating.csv', sep='\t', names=['user', 'item', 'rate', 'time'])
msk = numpy.random.rand(len(df)) < 0.7 # массив Истина или Ложь, где Истина будет с 70% вероятностью
# df_train = df[msk] # содержит 70% от всех данных (для тренировки)

df_train = df

num_users = len(pd.Series(df_train.user).unique())
num_items = len(pd.Series(df_train.item).unique())

user_indecies = [x-1 for x in df_train.user.values] # номера пользователей идут с 1, надо превратить их в индексы массива
item_indecies = [x-1 for x in df_train.item.values] # аналогично пользователям
rates = df_train.rate.values  # колонка оценок

# размер вектора оценок
# вектор оценок нужен для того, чтобы на каждом шаге не перемножать полные матрицы оценок
# "оценки пользователей по фильмам" на "оценки от пользователей фильмам"
feature_len = 10

# U = tf.truncated_normal([num_users, feature_len])
# with tf.Session() as sess:
#	u = sess.run([U])
#	tf.Print(u,[u])
#sys.exit()


# каждому пользователю сопоставляем вектор оценок размером feature_len, которые поставил этот пользователь (инициализируем случайными числами)
U = tf.Variable(initial_value=tf.truncated_normal([num_users, feature_len]), name='users')

# каждому фильму сопоставляем вектор оценок размером feature_len, которые поставили этому фильму (инициализируем случайными числами)
P = tf.Variable(initial_value=tf.truncated_normal([feature_len, num_items]), name='items')

# перемножаем матрицу оценок пользователей на матрицу оценок фильмов
# получаем матрицу оценки каждого пользователя по каждому фильму
result = tf.matmul(U, P) 

# вытяенем матрицу оценок в одномерный вектор
result_flatten = tf.reshape(result, [-1])

# T = user_indecies * tf.shape(result)[1] + item_indecies
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
# t = sess.run(T)
# print(t)
# sys.exit()

# т.к. мы вытянули матрицу оценок (пользователи в строках, фильмы в столбцах) в плоский вектор, то для того, чтобы узнать какую оценку
# поставил пользователь фильму надо индекс пользователя (номер строки матрицы, начиная с 0) умножить на кол-во фильмов
# так мы перескочим на начало оценок данного пользователя
# затем добавить индекс фильма
# и окажемся на ячейке оценки фильма пользователем
R = tf.gather(result_flatten, user_indecies * tf.shape(result)[1] + item_indecies, name='extracting_user_rate')

# РАСЧЕТ ОШИБКИ

# вектор разницы между реальными оценками rates и расчитываемыми R
diff_op = tf.subtract(R, rates, name='trainig_diff')

# модуль вектора ошибок
diff_op_squared = tf.abs(diff_op, name="squared_difference")

# суммируем вектор ошибки по всем оценкам в одну итоговую сумму
base_cost = tf.reduce_sum(diff_op_squared, name="sum_squared_error")

# regularization
lda = tf.constant(.001, name='lambda')
norm_sums = tf.add(tf.reduce_sum(tf.abs(U, name='user_abs'), name='user_norm'),
                   tf.reduce_sum(tf.abs(P, name='item_abs'), name='item_norm'))
regularizer = tf.multiply(norm_sums, lda, 'regularizer')

cost = tf.add(base_cost, regularizer)

# ОБУЧЕНИЕ

# скорость обучения на начальном этапе
lr = tf.constant(.001, name='learning_rate')

# скорость обучения будет меняться каждый раз по прохождению 10000 шагов см. exponential_decay (адаптивная скорость обучения)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)

# минимизация ошибки (cost) по алгоритму градиентного спуска
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(cost, global_step=global_step)

# запуск обучения на 1000 эпох
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    sess.run(training_step)


# calculate accuracy
# df_test = df[~msk]
# user_indecies_test = [x-1 for x in df_test.user.values]
# item_indecies_test = [x-1 for x in df_test.item.values]
# rates_test = df_test.rate.values

# accuracy
# R_test = tf.gather(result_flatten, user_indecies_test * tf.shape(result)[1] + item_indecies_test, name='extracting_user_rate_test')
# r = sess.run(R_test)
# print(r)

# diff_op_test = tf.subtract(R_test, rates_test, name='test_diff')
# diff_op_squared_test = tf.abs(diff_op, name="squared_difference_test")
# cost_test = tf.divide(tf.reduce_sum(tf.square(diff_op_squared_test, name="squared_difference_test"), name="sum_squared_error_test"), tf.cast(df_test.shape[0] * 2, tf.float32), name="average_error")
# print(sess.run(cost_test))

# builder = tf.saved_model.builder.SavedModelBuilder("~/tf/model")

# builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
#         "model": tf.saved_model.signature_def_utils.predict_signature_def(
#             inputs= {"x": x},
#             outputs= {"finalnode": model})
#         })
# builder.save()

# converter = tf.lite.TFLiteConverter.from_saved_model("~/tf/model")
# tflite_model = converter.convert()
# open("~/tf/cf_model.tflite", "wb").write(tflite_model)