import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

df.pop('id')
df = df.dropna()

df1 = df.loc[df['class'] == 1]
df2 = df.loc[df['class'] == 0][:int(len(df1)*1)] 

df = pd.concat([df1,df2])

target = df.pop('class')
df = df.to_numpy()
target = target.to_numpy()
xtrain, xtest,ytrain ,ytest = train_test_split(df, target, test_size = 0.2, random_state = 42)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64,input_dim=64, activation='sigmoid'),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=25)

test_loss, test_acc = model.evaluate(xtest, ytest, verbose=2)

a = np.argmax(model.predict(xtest), axis=-1)

print(model.summary())

model.save('modelis')

