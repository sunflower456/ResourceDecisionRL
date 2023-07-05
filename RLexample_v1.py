# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./input"))
print(os.listdir("./working"))
# Any results you write to the current directory are saved as output.

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 # stay, watching, up, down
		self.memory = deque(maxlen=1000)
		self.metric = []
		self.watching_count = 0
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model("./working/" + model_name) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def decide_action(self, state):
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		options = self.model.predict(state)
		return np.argmax(options[0])
		
	def act(self, action, request, data):
		reward = 0

		if action == 0: #stay
			reward = max(request - data, 0)
			print("\n[Stay] Reward: "+str(reward)+" CPU: "+str(round(float(data), 2)))
		elif action == 1: # watching
			self.metric.append(data)
			if len(self.metric) >= 0 and len(self.metric) <= 3: # TO BE CHANGED
				reward = max(request - data, 0)
			print("\n[Watching] Reward: "+str(reward)+" CPU: "+str(round(float(data), 2)))
		elif action == 2: # down
			if len(self.metric) > 3:
				reward = max(request - data, 0)
			self.metric.clear()
			print("\n[Down] Reward: "+str(reward)+" CPU: "+str(round(float(data), 2)))
		elif action == 3: # up
			if len(self.metric) == 0:
				reward = max(data - request, 0)
			self.metric.clear()
			print("\n[Up] Reward: "+str(reward)+" CPU: "+str(round(float(data), 2)))

		return reward			
			
	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("./input/"+key+".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(round(float(line.split(",")[3]) / 10000000, 2))

	return vec

def getRequestDataVec(key):
	vec = []
	lines = open("./input/"+key+".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(round(float(line.split(",")[2]), 2))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))
	return np.array([res])

def plot_reward(total_rewards, filename):
    plt.figure(figsize=(10,6)) 
    plt.plot(total_rewards, label='reward') 
    plt.title('total_rewards')
    plt.legend()
    plt.savefig(filename+".png") 


def train():
    stock_name, window_size, episode_count = 'kubernetes_pod_container_behavior_20230531', 12, 20

    agent = Agent(window_size)
    data = getStockDataVec(stock_name)
    request = 2
    l = len(data) - 1
    batch_size = 32
    total_rewards = []
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        agent.metric = []
        state = getState(data, 0, window_size + 1)
        total_reward = 0
        for t in range(l):
            action = agent.decide_action(state)
            reward = 0
            next_state = getState(data, t + 1, window_size + 1)
            reward = agent.act(action, request, data[t])
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state    
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
		
        total_rewards.append(total_reward)    
        if e % 10 == 0:
            agent.model.save("./working/model_ep" + str(e))
            plot_reward(total_rewards, "reward_"+str(e))
    return total_rewards
			
# def evaluation():
#     total_rewards = []
#     total_reward = 0
#     stock_name = "GSPC_2011"
#     model_name = "DQN_ep10.h5"
#     model = load_model("./working/"+model_name)
#     window_size = model.layers[0].input.shape.as_list()[1]
#     agent = Agent(window_size, True, model_name)
#     data = getStockDataVec(stock_name)
#     l = len(data) - 1
#     batch_size = 32
#     state = getState(data, 0, window_size + 1)
#     total_profit = 0
#     total_reward = 0
#     for t in range(l):
#         action = agent.decide_action(state)
#         print(action)
#         # sit
#         next_state = getState(data, t + 1, window_size + 1)
#         reward = 0
#         if action == 1: # buy
#             agent.inventory.append(data[t])
#             print("Buy: " + formatPrice(data[t]))
#         elif action == 2 and len(agent.inventory) > 0: # sell
#             bought_price = agent.inventory.pop(0)
#             reward = max(data[t] - bought_price, 0)
#             total_profit += data[t] - bought_price
#             print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
#         done = True if t == l - 1 else False
#         agent.memory.append((state, action, reward, next_state, done))
#         state = next_state

#         total_reward += reward
#         total_rewards.append(total_reward)

#         if done:
#             print("--------------------------------")
#             print(stock_name + " Total Profit: " + formatPrice(total_profit))
#             print("--------------------------------")
#             print ("Total profit is:",formatPrice(total_profit))
#     return total_rewards        

total_rewards = train()
plot_reward(total_rewards, "final")