import os

import pandas as pd
from keras.models import load_model, Model

import sqlite3

path = os.path.dirname(os.path.abspath(__file__))

# восстановим индекс товаров
df = pd.read_csv(os.path.join(path, "item_idx.csv"), names=['id', 'value'], sep='\t')
item_idx = df.set_index('id')['value'].to_dict()

df = pd.read_csv(os.path.join(path, "rating.csv"), sep='\t')
df.head()

# модель обучали по индексам, поэтому нам достаточно узнать сколько покупателей и сколько товаров
num_customers = len(df.customer.unique())
num_items = len(df.item.unique())

# полный перебор каждого покупателя по каждому товару
data = [] 
for customer in range(num_customers):
	for item in range(num_items):
		data.append([customer, item])

df = pd.DataFrame(data, columns = ['customer', 'item']) 

# получаем предсказания оценок каждого покупателя по каждому товару
model = load_model(os.path.join(path, "cf_model.h5"))
rates = model.predict([df.customer, df.item])

# запишем это все в базу данных
conn = sqlite3.connect(os.path.join(path, "rates.db"))
cursor = conn.cursor()
 
cursor.execute("DROP TABLE IF EXISTS rates")
cursor.execute("CREATE TABLE IF NOT EXISTS rates(customer integer, item integer, rate real)")

for customer in range(num_customers):

	rows = []
	for item in range(num_items):
		rows.append((customer, item_idx[item], float(rates[customer * num_items + item])))

	cursor.executemany("INSERT INTO rates VALUES(?,?,?)", rows)
	conn.commit()
