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
        self.action_size = 3 # stay, watching, up, down
        self.memory = deque(maxlen=1000)
        self.metric = []
        self.scale_size = 0
        self.request = 3.0
        self.num_steps = 1
        self.num_features = state_size

        self.seq = []
        self.watching_count = 0
        self.model_name = model_name
        self.is_eval = is_eval
        self.net == 'dnn'
        self.value_network = self.init_value_network()
        self.policy_network = self.init_policy_network()
        self.value_network_path = None
        self.policy_network_path = None

        self.discount_factor = 0.9
        self.epsilon = 1.0
        self.lr = 0.0005
        self.epsilon_decay = 0.995

        self.model = load_model("./working/" + model_name) if is_eval else self._model()

    def reset(self, data, t, n):
        seq = create_sequences(data, n)
        self.seq = seq
        self.state = seq[0]
        self.metric = []
        self.scale_size = 0
        self.request = 3.0
        return np.array(self.state, dtype=np.float32)
        
    def decide_action(self, pred_value, pred_policy, epsilon):
       
        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.action_size)
        else:
            exploration = False
            action = np.argmax(pred)

        return action
		
    def act(self, action, request, t, n):
        reward = 0
        dataLen = len(data)

        if t + n >= dataLen:
            seq = self.seq[dataLen-n-1]
            next_seq = self.seq[dataLen-n-1]
        else:
            seq = self.seq[t]
            next_seq = self.seq[t+1]

        for i in range(len(seq)):
            tempList.append(seq[i] - request)
            
        if action == 0: #down
            if tempList[0] > 0:
                reward = -0.5
        elif action == 1: # up
            if tempList[0] > 0:
                reward = 1
            else:
                reward = -1
        elif action == 2: # stay or watching
            if tempList[0] < 0:
                if (tempList[1] < 0) & (tempList[2] < 0)& (tempList[2] < 0) :
                    reward = 1

        return reward, next_seq			
			
    def get_batch(self):
            memory = self.memory
            x = np.zeros((len(self.memory), self.num_steps, self.num_features))
            y_value = np.zeros((len(self.memory), self.action_size))
            value_max_next = 0
            for i, (sample, next_sample, action, reward) in enumerate(memory):
                x[i] = sample
                r = self.memory[-1] - reward
                y_value[i] = value
                y_value[i, action] = r + self.discount_factor * value_max_next
                value_max_next = value.max()
            return x, y_value, None

    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss
            

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.action_size, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.action_size, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.action_size, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid', 
                            loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.action_size, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.action_size, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.action_size, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)


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


def plot_reward(total_rewards, filename):
    plt.figure(figsize=(10,6)) 
    plt.plot(total_rewards, label='reward') 
    plt.title('total_rewards')
    plt.legend()
    plt.savefig(filename+".png") 

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        xs.append(x)
    # random.shuffle(xs)    
    return np.array(xs)

def train():
    stock_name, window_size, episode_count = 'kubernetes_pod_container_portal_20230624', 4, 5000

    agent = Agent(window_size)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    start_epsilon = 1
    total_rewards = []
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        sample = agent.reset(data, 0, window_size + 1)
        total_reward = 0
        epsilon = start_epsilon * (1 - (e / (episode_count)))
        for t in range(l):

            pred_value = None
            pred_policy = None
            
            pred_value = self.value_network.predict(sample)
            pred_policy = self.policy_network.predict(sample)
            
            action = agent.decide_action(pred_value, pred_policy, epsilon)
            reward, next_sample = agent.act(action, request, t, window_size)
            # done = True if t == l - 1 else False
            agent.memory.append((sample, next_sample, action, reward))
            sample = next_sample
            total_reward += reward
        
        self.fit()
        total_rewards.append(total_reward)    
        if e % 10 == 0:
            agent.model.save("./working/model_ep" + str(e))
            plot_reward(total_rewards, "reward_"+str(e))
    return total_rewards
		      

train()
