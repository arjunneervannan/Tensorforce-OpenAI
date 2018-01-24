from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import matplotlib as mpl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning-rate', help="Learning Rate")

args = parser.parse_args()

learning_rate = 0

if args.learning_rate is not None:
	learning_rate = args.learning_rate

os.chdir(os.path.join(r'c:\users\rneervannan\Desktop\Results-2018', str(learning_rate)))

trpo_results = list()
ppo_results = list()
vpg_results = list()

def csvopen(file):
	end_results = list()
	with open(file) as f:
		reader = csv.reader(f)
		for row in reader:
			end_results.append(float(row[1]))
		return end_results
def meancalculate(start_result):
	mean = list()
	for i in range(len(start_result)):
		mean.append(np.mean(start_result[i:i+1000]))
	return mean

# print("Last 100 episodes mean for TRPO, PPO, and VPG:", np.mean(csvopen('trpo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000], 
# 	np.mean(csvopen('ppo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000], 
# 	np.mean(csvopen('vpg_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000])

print("Average Reward from last 100 episodes for TRPO, PPO, and VPG:")
print(np.mean((csvopen('trpo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000]),
	np.mean((csvopen('ppo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000]),
	np.mean((csvopen('vpg_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'))[39900:40000]))

mpl.style.use('default')
VPG, = plt.plot(csvopen('vpg_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'), color=(0.675, 0.651, 1), alpha=0.5, label='VPG')
TRPO, = plt.plot(csvopen('trpo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'), color=(1, 0.675, 0.651), alpha=0.5, label='TRPO')
PPO, = plt.plot(csvopen('ppo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv'), color=(0.675, 1, 0.651), alpha=0.5, label='PPO')
VPGavg, = plt.plot(meancalculate(csvopen('vpg_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv')), color=(0, 0, 1), label='VPG Average', alpha=0.7)
TRPOavg, = plt.plot(meancalculate(csvopen('trpo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv')), color=(1, 0, 0), label='TRPO Average', alpha=0.7)
PPOavg, = plt.plot(meancalculate(csvopen('ppo_resultsHalfCheetah-v1_' + str(learning_rate) + '.csv')), color=(0, 1, 0.5), label='PPO Average', alpha=0.7)

plt.legend(handles=[TRPOavg, PPOavg, VPGavg, VPG, TRPO, PPO])
plt.xlabel('Number of Episodes')
plt.ylabel('Total Reward per Episode')
plt.title('Reward per Episode vs. Number of Episodes for Learning Rate ' + str(learning_rate))
plt.show()
