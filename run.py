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