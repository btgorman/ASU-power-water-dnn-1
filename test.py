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
import matplotlib.pyplot as plt

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

APPEND_AS_COLUMNS = 1
APPEND_AS_ROWS = 0

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

columns = list(output_tensor.columns)

length = int(len(columns)*0.1)
begin = 2*length
end = min(begin + length, len(columns))
for col in range(begin, end):
	plt.hist(output_tensor[columns[col]], bins=60)
	plt.show()