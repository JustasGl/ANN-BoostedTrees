import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

MaxSteps = 100
df = pd.read_csv('data.csv')

df.pop('id')
df = df.dropna()

df1 = df.loc[df['class'] == 1]
df2 = df.loc[df['class'] == 0][:int(len(df1)*1)] 

df = pd.concat([df1,df2])

target = df.pop('class')
target = target.to_numpy()

dftrain, dfeval,y_train ,y_eval = train_test_split(df, target, test_size = 0.2, random_state = 42)

NUMERIC_COLUMNS = ['net profit / total assets','total liabilities / total assets',
'working capital / total assets','current assets / short-term liabilities',
'[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365',
'retained earnings / total assets','EBIT / total assets','book value of equity / total liabilities',
'sales / total assets','equity / total assets','(gross profit + extraordinary items + financial expenses) / total assets',
'gross profit / short-term liabilities','(gross profit + depreciation) / sales','(gross profit + interest) / total assets',
'(total liabilities * 365) / (gross profit + depreciation)','(gross profit + depreciation) / total liabilities',
'total assets / total liabilities','gross profit / total assets','gross profit / sales','(inventory * 365) / sales',
'sales (n) / sales (n-1)','profit on operating activities / total assets','net profit / sales',
'gross profit (in 3 years) / total assets','(equity - share capital) / total assets','(net profit + depreciation) / total liabilities',
'profit on operating activities / financial expenses','working capital / fixed assets','logarithm of total assets',
'(total liabilities - cash) / sales','(gross profit + interest) / sales','(current liabilities * 365) / cost of products sold',
'operating expenses / short-term liabilities','operating expenses / total liabilities','profit on sales / total assets',
'total sales / total assets','(current assets - inventories) / long-term liabilities','constant capital / total assets',
'profit on sales / sales','(current assets - inventory - receivables) / short-term liabilities',
'total liabilities / ((profit on operating activities + depreciation) * (12/365))','profit on operating activities / sales',
'rotation receivables + inventory turnover in days','(receivables * 365) / sales','net profit / inventory',
'(current assets - inventory) / short-term liabilities','(inventory * 365) / cost of products sold',
'EBITDA (profit on operating activities - depreciation) / total assets','EBITDA (profit on operating activities - depreciation) / sales',
'current assets / total liabilities','short-term liabilities / total assets','(short-term liabilities * 365) / cost of products sold)',
'equity / fixed assets','constant capital / fixed assets','working capital','(sales - cost of products sold) / sales',
'(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)','total costs /total sales',
'long-term liabilities / equity','sales / inventory','sales / receivables','(short-term liabilities *365) / sales',
'sales / short-term liabilities','sales / fixed assets']

feature_columns = []
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                           dtype=tf.float32))

NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps=MaxSteps)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)
print(pd.Series(result))

n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=MaxSteps)

# Eval.
result = est.evaluate(eval_input_fn)
print(pd.Series(result))

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
#tf.keras.experimental.export_saved_model(est, 'ModelisTree')
#est.export_saved_model('ModelisTree')

probs.plot(kind='hist', bins=20, title='predicted probabilities')

a = pd.Series([pred['class_ids'][0] for pred in pred_dicts])
cnt = 0
for i in range(len(a)):
    if a[i] == y_eval[i]:
        cnt+=1
    print(a[i], ' = ', y_eval[i])

print('Accuracy: ', cnt/len(a))

plt.show()

