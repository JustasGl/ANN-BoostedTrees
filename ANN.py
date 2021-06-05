import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_dim=64, activation='sigmoid'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(training, TrainTarget, epochs=25)

test_loss, test_acc = model.evaluate(testing, Testtarget, verbose=2)

a = model.predict(testing)


fpr_keras, tpr_keras, thresholds_keras = roc_curve(Testtarget, a)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

print(model.summary())

a[a > auc_keras] = 1
a[a < auc_keras] = 0

con_mat = tf.math.confusion_matrix(labels=Testtarget, predictions=a).numpy()

print('ANN Confusion matric')

print(con_mat)

model.save('modelis')



