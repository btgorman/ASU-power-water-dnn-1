# Copyright 2017 Brandon Tom Gorman

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import subprocess
import sys

number_of_layers = 5
# opt_list = ['Adadelta', 'Adam', 'Adamax', 'Nadam', 'RMSprop']
opt_list = ['Adam']
batch_list = [128, 64, 32]
for i in range(0, len(opt_list)):
	for j in range(number_of_layers, number_of_layers+1):
		for l in range(0,5):
			for k in range(0, len(batch_list)):
				pid = subprocess.call('python mlearn.py {} {} {}'.format(j, opt_list[i], batch_list[k]), shell=True)