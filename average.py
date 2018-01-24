import numpy as np
import csv
import os
from numpy import arange
from pandas import *

learning_rate = 0

def csvopen(file):
	end_results = list()
	with open(file) as f:
		reader = csv.reader(f)
		for row in reader:
			end_results.append(float(row[1]))
		return end_results

# np.mean((csvopen('trpo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000])
# np.mean((csvopen('ppo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000])
# np.mean((csvopen('vpg_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000])


for learning_rate in [0.1, 0.01, 0.001, 0.0001, 0.0003, '0.00001']:
	print learning_rate
	os.chdir(os.path.join(r'c:\users\rneervannan\Desktop\Results-2018', str(learning_rate)))

	print("Average Reward from last 100 episodes for TRPO, PPO, and VPG and LR:", learning_rate)
	print(np.mean((csvopen('trpo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000]),
		np.mean((csvopen('ppo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000]),
		np.mean((csvopen('vpg_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000]))
