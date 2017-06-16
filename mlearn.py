# Copyright 2017 Brandon Tom Gorman

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import pandas as pd
import numpy as np
from sklearn import preprocessing
import csv
import sys

from keras.models import Sequential, optimizers
from keras.layers import Dense, Dropout, Activation, advanced_activations, normalization
from keras import callbacks, losses, backend

input_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
output_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

APPEND_AS_COLUMNS = 1
APPEND_AS_ROWS = 0

INPUT_LAYER_DIM = 0
OUTPUT_LAYER_DIM = 0

number_of_epochs = 210
learn_rate = 0.001
decay_rate = 0.000255
loss_select = 'mse'
hl_iteration = int(sys.argv[1])
opt_select = str(sys.argv[2])
batch_select = int(sys.argv[3])
opt_iteration = 3

print(opt_select, str(hl_iteration) + '_' + str(opt_iteration) + '_', batch_select)

class LossHistory(callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.val_losses = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))

def max_absolute_error(y_true, y_pred):
	return backend.max(backend.abs(y_pred - y_true), axis=-1)

def adjust_learn(inc):
	return learn_rate * (1.0 / (1.0 + decay_rate*inc))

# TRAINING AND VALIDATION DATA SET
input_tensor_continuous_1 = pd.read_csv('./1/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_2 = pd.read_csv('./2/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_3 = pd.read_csv('./3/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_4 = pd.read_csv('./4/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_5 = pd.read_csv('./5/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_6 = pd.read_csv('./6/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_7 = pd.read_csv('./7/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_8 = pd.read_csv('./8/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_9 = pd.read_csv('./9/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_10 = pd.read_csv('./10/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_11 = pd.read_csv('./11/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_12 = pd.read_csv('./12/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)

input_tensor_categorical_1 = pd.read_csv('./1/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_2 = pd.read_csv('./2/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_3 = pd.read_csv('./3/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_4 = pd.read_csv('./4/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_5 = pd.read_csv('./5/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_6 = pd.read_csv('./6/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_7 = pd.read_csv('./7/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_8 = pd.read_csv('./8/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_9 = pd.read_csv('./9/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_10 = pd.read_csv('./10/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_11 = pd.read_csv('./11/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_12 = pd.read_csv('./12/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)

output_tensor_1 = pd.read_csv('./1/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_2 = pd.read_csv('./2/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_3 = pd.read_csv('./3/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_4 = pd.read_csv('./4/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_5 = pd.read_csv('./5/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_6 = pd.read_csv('./6/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_7 = pd.read_csv('./7/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_8 = pd.read_csv('./8/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_9 = pd.read_csv('./9/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_10 = pd.read_csv('./10/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_11 = pd.read_csv('./11/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_12 = pd.read_csv('./12/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)

input_tensor_continuous = np.concatenate((input_tensor_continuous_1.values, input_tensor_continuous_2.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_3.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_4.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_5.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_6.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_7.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_8.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_9.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_10.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_11.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = np.concatenate((input_tensor_continuous, input_tensor_continuous_12.values), axis=APPEND_AS_ROWS)
input_tensor_continuous = input_scaler.fit_transform(input_tensor_continuous)

input_tensor_categorical = np.concatenate((input_tensor_categorical_1.values, input_tensor_categorical_2.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_3.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_4.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_5.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_6.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_7.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_8.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_9.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_10.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_11.values), axis=APPEND_AS_ROWS)
input_tensor_categorical = np.concatenate((input_tensor_categorical, input_tensor_categorical_12.values), axis=APPEND_AS_ROWS)

input_tensor = np.concatenate((input_tensor_continuous, input_tensor_categorical), axis=APPEND_AS_COLUMNS)

output_tensor = np.concatenate((output_tensor_1.values, output_tensor_2.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_3.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_4.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_5.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_6.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_7.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_8.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_9.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_10.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_11.values), axis=APPEND_AS_ROWS)
output_tensor = np.concatenate((output_tensor, output_tensor_12.values), axis=APPEND_AS_ROWS)

output_tensor = pd.DataFrame(output_tensor)
output_tensor_drop_cols = output_tensor.std()[output_tensor.std() < .2].index.values
output_tensor = output_tensor.drop(output_tensor_drop_cols, axis=1)
output_tensor = output_tensor.values

output_tensor = output_scaler.fit_transform(output_tensor)

input_tensor_continuous_tests = pd.read_csv('./13/input_tensor_continuous.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_continuous_tests = input_tensor_continuous_tests.values
input_tensor_continuous_tests = input_scaler.transform(input_tensor_continuous_tests)
input_tensor_categorical_tests = pd.read_csv('./13/input_tensor_categorical.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
input_tensor_categorical_tests = input_tensor_categorical_tests.values
input_tensor_tests = np.concatenate((input_tensor_continuous_tests, input_tensor_categorical_tests), axis=APPEND_AS_COLUMNS)

output_tensor_tests = pd.read_csv('./13/output_tensor.csv', sep=' ', header=None, index_col=None, dtype=np.float64)
output_tensor_tests = output_tensor_tests.drop(output_tensor_drop_cols, axis=1)
output_tensor_tests = output_tensor_tests.values
output_tensor_tests = output_scaler.transform(output_tensor_tests)

#LAYER DIMENSIONS
_, INPUT_LAYER_DIM = input_tensor.shape
_, OUTPUT_LAYER_DIM = output_tensor.shape

''' NEURAL NETWORK MODEL '''
drop_rate = 0.0
model = Sequential()

model.add(Dense(OUTPUT_LAYER_DIM*12, input_shape=(INPUT_LAYER_DIM, ), kernel_initializer='he_uniform')) #1
model.add(normalization.BatchNormalization())
model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 2:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM*9, kernel_initializer='he_uniform')) #2
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 3:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM*6, kernel_initializer='he_uniform')) #3
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 4:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM*3, kernel_initializer='he_uniform')) #4
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 5:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM, kernel_initializer='he_uniform')) #5
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 6:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM, kernel_initializer='he_uniform')) #6
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 7:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM, kernel_initializer='he_uniform')) #7
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))
if hl_iteration == 8:
	# model.add(Dropout(drop_rate))
	model.add(Dense(OUTPUT_LAYER_DIM, kernel_initializer='he_uniform')) #8
	model.add(normalization.BatchNormalization())
	model.add(advanced_activations.LeakyReLU(alpha=0.2))

history = LossHistory()
learn = callbacks.LearningRateScheduler(adjust_learn)
saver = callbacks.ModelCheckpoint('./mloutputs/' + opt_select + '/' + str(batch_select) + '/' + str(hl_iteration) + '_' + str(opt_iteration) + '_model_{epoch:02d}_{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True, period=209)
# stopper = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=0, mode='auto')

model.compile(optimizer=opt_select, loss=loss_select, metrics=['mae'])
model.fit(input_tensor, output_tensor, validation_split=0.0, epochs=number_of_epochs, batch_size=batch_select, callbacks=[history, learn, saver])

predict_input = input_tensor[-2:-1, :]
predict_true = output_tensor[-2:-1, :]
predict_vals = model.predict(predict_input, batch_size=batch_select)

predict_true = output_scaler.inverse_transform(predict_true)
predict_vals = output_scaler.inverse_transform(predict_vals)

with open('./mloutputs/' + opt_select + '/' + str(hl_iteration) + '_' + str(opt_iteration) + '.csv', 'w') as csvfile:
	csvoutput = csv.writer(csvfile, delimiter=',')
	csvoutput.writerow(history.losses)
	csvoutput.writerow(history.val_losses)

with open('./mloutputs/' + opt_select + '/' + str(hl_iteration) + '_' + str(opt_iteration) + 'prediction' + '.csv', 'w') as csvfile:
	csvoutput = csv.writer(csvfile, delimiter=',')
	csvoutput.writerow(predict_true[0, :])
	csvoutput.writerow(predict_vals[0, :])