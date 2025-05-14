import pandas as pd
import numpy as np

df = pd.read_csv('dataset/kc_house_data.csv')
df = df.drop('id', axis=1)
y = df['price']
# df['date'] = pd.to_datetime(df['date'])
# df['month'] = df['date'].apply(lambda date: date.month)
# df['year'] = df['date'].apply(lambda date: date.year)
df.drop('price',axis=1,inplace=True)
df.drop('date',axis=1,inplace=True)
y.to_csv('dataset/y.csv')
df.to_csv('dataset/x.csv')
print(df.head())