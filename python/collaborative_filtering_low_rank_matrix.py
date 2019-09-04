# за основу взят пример:
# https://www.kaggle.com/rajmehra03/cf-based-recsys-by-low-rank-matrix-factorization

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

import keras
from keras.optimizers import Adam
from keras.layers.merge import dot
from keras.layers import Flatten, Input, Embedding


def get_model():

    # входные данные покупеталей
    customer_input = Input(shape=(1,), name='customer_input', dtype='int64')
    # каждому покупателю сопоставляем вектор коэффициентов размером n_latent_factors (строим матрицу "П")
    customer_embedding = Embedding(n_customers, n_latent_factors, name='customer_embedding')(customer_input)
    # Embedding возвращает трехмерную матрицу размерностью (n_customers, 1, n_latent_factors)
    # необходимо привести ее в двумерному виду (n_customers, n_latent_factors)
    customer_vec = Flatten(name='FlattenCustomers')(customer_embedding)

    # входные данные товаров
    item_input = Input(shape=(1,), name='item_input', dtype='int64')
    # каждому товару сопоставляем вектор коэффициентов размером n_latent_factors (строим матрицу "Т")
    item_embedding = Embedding(n_items, n_latent_factors, name='item_embedding')(item_input)
    # Embedding возвращает трехмерную матрицу размерностью (n_items, 1, n_latent_factors)
    # необходимо привести ее в двумерному виду (n_items, n_latent_factors)
    item_vec = Flatten(name='FlattenItems')(item_embedding)

    # матрица оценок О = П * Т
    sim = dot([customer_vec, item_vec], name='Simalarity-Dot-Product', axes=1) 
    #sim = K.dot(customer_vec, t_item_vec) # Dot(), name='Simalarity-Dot-Product', axes=1) 

    return keras.models.Model([customer_input, item_input], sim)


path = os.path.dirname(os.path.abspath(__file__))

# читаем данные из файла rating.csv
df = pd.read_csv(os.path.join(path, "rating.csv"), sep='\t')
df.head()

# уникальные покупатели и товары
customers = df.customer.unique()
items = df.item.unique()

# т.к. номера покупателей и товаров могут быть любыми, то создадим словарь
# где сопоставим текущий код покупателя/товара новому индексу от 0 до кол-ва покупателей/товаров
customer2idx = {o: i for i, o in enumerate(customers)}
item2idx = {o: i for i, o in enumerate(items)}

# заменим в данных коды товаров и покупателей на их индексы
df['customer'] = df['customer'].apply(lambda x: customer2idx[x])
df['item'] = df['item'].apply(lambda x: item2idx[x])

# сохраним индекс товаров
idxf = open(os.path.join(path, "item_idx.csv"), "w")
for k, v in item2idx.items():
    idxf.write(str(v) + '\t' + str(k) + '\n')
idxf.close()

# поделим данные на 80% тренировочных и 20% проверочных
split = np.random.rand(len(df)) < 0.8
train = df[split]
valid = df[~split]

# количество покупателей, количество товаров, размер вектора коэффициентов
n_items = len(df['item'].unique())
n_customers = len(df['customer'].unique())
n_latent_factors = 100

# модель
model = get_model()
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

# параметры обучения модели
batch_size = 128
epochs = 15

# учимся
History = model.fit([train.customer, train.item],
                    train.rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([valid.customer, valid.item], valid.rate),
                    verbose=1)

# сохраним модель, чтобы не учить потом, если она нам понадобится
model.save(os.path.join(path, "cf_model.h5"))

# график обучения (для наглядности)

rcParams['figure.figsize'] = 10, 5

plt.plot(History.history['loss'], 'g')
plt.plot(History.history['val_loss'], 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(path, "train.png"))
