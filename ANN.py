import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PrintRoc import PrintRoc 
from ReadnFormat import ReadnFormat

training, testing, TrainTarget, Testtarget = ReadnFormat(TestingDataPercent = 0.2, SamplingPercentageOfBancrupt = 2, ConvertToNp = 1)

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

Predicted = model.predict(testing)

auc_keras,_,_,_ = PrintRoc(Predicted=Predicted, Testtarget = Testtarget)

print(model.summary())


Predicted[Predicted > auc_keras] = 1
Predicted[Predicted < auc_keras] = 0

con_mat = tf.math.confusion_matrix(labels=Testtarget, predictions=Predicted).numpy()

print('ANN Confusion matric')

print(con_mat)

model.save('modelis')



