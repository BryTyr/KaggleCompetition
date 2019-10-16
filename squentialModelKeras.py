from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
import numpy as np
import seaborn as seabornInstance
import pandas_profiling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def read_CSV_file(filename):
    return pd.read_csv(filename)

def LabelEncoder_data(data,row):
    le = LabelEncoder()
    le.fit(data[row])
    data[row]=le.transform(data[row])

def replace_bad_data(data,column,replacedValue):
    data[column]= data[column].replace("0", replacedValue)
    data[column]= data[column].replace("#N/A", replacedValue)
    data[column]= data[column].replace(0, replacedValue)
    data[column]= data[column].replace("nan", replacedValue)
    data[column]= data[column].replace("", replacedValue)
    data[column]= data[column].replace(" ", replacedValue)
    data[column]= data[column].replace(np.nan, replacedValue)
    data[column]= data[column].replace("unknown", replacedValue)


def set_row_datatype(data):
    data=data.astype({'Gender': 'str'})
    data=data.astype({'University Degree': 'str'})
    data=data.astype({'Year of Record': 'int'})
    data=data.astype({'Country': 'str'})
    data=data.astype({'Hair Color': 'str'})
    data=data.astype({'Profession': 'str'})
    if "Income in EUR" in data.columns:
        data=data.astype({'Income in EUR': 'int'})


def printResults(test_targets, test_predictions):
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_predictions)))


def one_hot_encoder(data,column):
    one_hot = pd.get_dummies(data[column])
    data = data.drop(column,axis = 'columns')
    return pd.concat([data,one_hot],axis='columns')


def build_model(training_dataset):
    # Sequential model used with two dense layers of 64 nodes
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(training_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # output layer
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.0005)
  model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
  return model


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Validation Error')
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Validation Error')
  plt.legend()
  plt.show()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation Error')
  plt.legend()
  plt.show()





#read training file and data file
dataset=read_CSV_file("tcd ml 2019-20 income prediction training (with labels).csv")
testset_kaggle=read_CSV_file("tcd ml 2019-20 income prediction test (without labels).csv")

# print out a visual report of the preprocessed data
dataset.profile_report(title="initData").to_file("initData.html")

#removes index column and income column for test data
dataset=dataset.drop("Instance", axis='columns')
testset_kaggle=testset_kaggle.drop("Instance", axis='columns')
testset_kaggle=testset_kaggle.drop("Income", axis='columns')

#back fill empty cells for year of record column
dataset.replace({'gain_sum_y': {0: np.nan}}).ffill()
testset_kaggle.replace({'gain_sum_y': {0: np.nan}}).ffill()

# get mean of age column
AgeMean=dataset["Age"].mean()
replace_bad_data(dataset,"Age",AgeMean)

AgeMean=testset_kaggle["Age"].mean()
replace_bad_data(testset_kaggle,"Age",AgeMean)

#remove invalid data from gender column
replace_bad_data(dataset,"Gender","unknownGen")
replace_bad_data(testset_kaggle,"Gender","unknownGen")

#remove invalid data from university degree column
replace_bad_data(dataset,"University Degree","unknownDeg")
replace_bad_data(testset_kaggle,"University Degree","unknownDeg")

#remove invalid data from profession column
replace_bad_data(dataset,"Profession","unknownPro")
replace_bad_data(testset_kaggle,"Profession","unknownPro")

#remove invalid data from profession column
replace_bad_data(dataset,"Hair Color","unknownHair")
replace_bad_data(testset_kaggle,"Hair Color","unknownHair")

replace_bad_data(dataset,"Year of Record",np.nan)
dataset.dropna(subset=['Year of Record'], inplace=True)

# sets the datatype for a particular set of data
set_row_datatype(dataset)
set_row_datatype(testset_kaggle)

# add a training data column so test and training data can be combined
# for one hot encoding
dataset['train']=1
testset_kaggle['train']=0

# country and profession are the two most important columns
#remove groups below a minimum threshold to improve relevance of data
# will also minimise extra features during one hot encoding
dataset=dataset.groupby('Country').filter(lambda x : len(x)>2)
dataset=dataset.groupby('Profession').filter(lambda x : len(x)>10)

#drop income in negitive not needed as income is also negitive in test data
#dataset = dataset[dataset['Income in EUR'] > 0]

# print out a visual report of the processed data  before one hot encoding
dataset.profile_report(title="initDataProcessed").to_file("initDataProcessed.html")

#Combine the dataset for one hot encoding
combined=pd.concat([dataset,testset_kaggle],axis=0,sort=False)

# drop gender column as it has 6.6% missing data and has very weak
# correlation to income label
# hair color dropped as it has no correlation to income in EUR
combined = combined.drop('Gender',axis='columns')
combined = combined.drop('Hair Color',axis='columns')
combined = combined.drop('Wears Glasses',axis='columns')

# One hot encode remaining catagorical values
combined=one_hot_encoder(combined,'Country')

combined=one_hot_encoder(combined,'Profession')

combined=one_hot_encoder(combined,'University Degree')

# now split the data bast to training and test data using the train column
dataset=combined.loc[combined['train'] == 1]
testset_kaggle=combined.loc[combined['train'] == 0]
dataset=dataset.drop("train", axis='columns')

#The testset has now got the income in Eur column from the combination and
# needs to be dropped
testset_kaggle=testset_kaggle.drop("train", axis='columns')
testset_kaggle=testset_kaggle.drop("Income in EUR", axis='columns')

# now with the initial data set have a 80/20 split between training and Testing
train_dataset = dataset.sample(frac=0.80,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#describes the data and gives good insights
train_stats = train_dataset.describe()
train_stats.pop("Income in EUR")
train_stats = train_stats.transpose()
print(train_stats)

# seperate the labls from the training data
train_labels = train_dataset.pop('Income in EUR')
test_labels = test_dataset.pop('Income in EUR')

# normalize certain columns listed below to give a fair weighting to all data
# it is done 3 times below for training testing and evaluation data
cols_to_norm = ['Year of Record','Age','Body Height [cm]','Size of City']

train_dataset[cols_to_norm] = train_dataset[cols_to_norm].apply(lambda x: (x - x.mean()) / ( x.std()))
normed_train_data=train_dataset

test_dataset[cols_to_norm] = test_dataset[cols_to_norm].apply(lambda x: (x - x.mean()) / ( x.std()))
normed_test_data=test_dataset

testset_kaggle[cols_to_norm] = testset_kaggle[cols_to_norm].apply(lambda x: (x - x.mean()) / ( x.std()))
normed_test_data_kaggle=testset_kaggle


# Display progress prints single dot each go over the data
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# number of itterations over the dataset
EPOCHS = 150

# Train the model
model = build_model(train_dataset)

# The patience parameter will check for loss and stop if loss becomes stagnant
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# fit the model
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0,batch_size=70, callbacks=[early_stop, PrintDot()])

# print the history of the model
plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error:"+mae)

test_predictions = model.predict(normed_test_data).flatten()

# create a scatter plot of the data
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()

# show the results of predicted vs actual labels
printResults(test_labels, test_predictions)

# save the evaluation results and submit to kaggle
test_prediction = model.predict(normed_test_data_kaggle)
print(test_prediction)
np.savetxt("output-model-predictions.csv", test_prediction, delimiter=",")
