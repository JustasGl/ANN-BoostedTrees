import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.pop('id')
df = df.dropna()#we marged all 5 years in one data file. And in data there were a lot of empty values were marked 
#with question mark sign Mantas converted them to null values and then we droped null values from dataframe

testing = df[0:int(len(df)*0.2)]
training = df[int(len(df)*0.2):]

df1 = training.loc[training['class'] == 1]
df2 = training.loc[training['class'] == 0][:int(len(df1)*1)] #because our data had inbalanced data. 
#Out of 43 thousand data lines there where 95 pecent of nonbankrupt companies and only 5 percent bancrupted how awfull is that right? Only around 5 % companies bancrupted.
# So with that in mind we searched for posible solutions. One of them was downsampling. SO we applied it. 
# #After downsampling there was 50/50 data distributions between bancrupt and non bancrupt companies

training = pd.concat([df1,df2])

Testtarget = testing.pop('class')
TrainTarget = training.pop('class')

testing = testing.to_numpy()
training = training.to_numpy()

Testtarget = Testtarget.to_numpy()
TrainTarget = TrainTarget.to_numpy()

training, testing,TrainTarget ,Testtarget = train_test_split(df, target, test_size = 0.2, random_state = 42)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64,input_dim=64, activation='sigmoid'),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(training, TrainTarget, epochs=25)

test_loss, test_acc = model.evaluate(testing, Testtarget, verbose=2)

a = np.argmax(model.predict(testing), axis=-1)

print(model.summary())

con_mat = tf.math.confusion_matrix(labels=Testtarget, predictions=a).numpy()

print(con_mat)

model.save('modelis')



