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
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability import distributions as tfd
from gym.utils import seeding
from gym import spaces, logger
import pylab

class ContinuousA2CEnv:
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.min_action = 0.0
		self.max_action = 10.0

		self.action_space = spaces.Box(
            low=-self.min_action,
            high=self.max_action,
            shape=(1,)
        )
		self.observation_space = spaces.Box(self.min_action, self.max_action)
		self.state = None

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self, data, t, n):
		self.state = getState(data, t, n)
		# self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		return np.array(self.state, dtype=np.float32)

	def step(self, action, resource):
		# self.state = getState(data, t, n)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		if action > resource:
			if resource / action * 100 < 80:
				reward = 1.0
			else:
				reward = 0.0
		elif action < resource:
			reward = -1
		else:
			reward = 0.0

		return np.array(self.state, dtype=np.float32), reward
	
class ContinuousA2C(tf.keras.Model):
    def __init__(self, action_size):
        super(ContinuousA2C, self).__init__()
        self.actor_fc1 = Dense(24, activation='tanh')
        self.actor_mu = Dense(action_size,
                              kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.actor_sigma = Dense(action_size, activation='sigmoid',
                                 kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1,
                                kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc1(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)
        sigma = sigma + 1e-5

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return mu, sigma, value

class ContinuousA2CAgent:
    def __init__(self, action_size, max_action, min_action):
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = ContinuousA2C(self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=1.0)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, self.min_action, self.max_action)
        return action

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            mu, sigma, value = self.model(state)
            _, _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            advantage = tf.stop_gradient(target - value[0])
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.1 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss, sigma

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

def plot_reward(score, loss, episode, filename):
    plt.subplot(2, 1, 1)  
    plt.plot(episode, score, 'b')
    plt.xlabel("episode")
    plt.title('score')
    plt.ylabel('average score')

    plt.subplot(2, 1, 2) 
    plt.plot(episode, loss, 'b')
    plt.xlabel("episode")
    plt.title('loss')
    plt.ylabel('average loss')

    plt.savefig(filename+".png")

def train():
    name, window_size, episode_count = 'kubernetes_pod_container_portal_20230624', 24, 200

    env = ContinuousA2CEnv()
    state_size = window_size
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]    
    agent = ContinuousA2CAgent(action_size, max_action, env.min_action)
    data = getStockDataVec(name)
    l = len(data) - 1
    total_rewards = []
    scores, mean_loss, episodes = [], [], []
    score_avg = 0
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        loss_list, sigma_list = [], []
        score = 0	
        state = env.reset(data, 0, window_size + 1)
        state = np.reshape(state, [1, state_size])
        for t in range(l):
            action = agent.get_action(state)
            next_state, reward = env.step(action, data[t])
            next_state = np.reshape(next_state, [1, state_size])
            print("action: ", action, " || data: ", data[t], " || reward: ", reward)
            score += reward	
            done = True if t == l - 1 else False
            loss, sigma = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            sigma_list.append(sigma)
            state = next_state
		
        total_rewards.append(score)   
        if done:
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f} | sigma: {:.3f}".format(e, score_avg, np.mean(loss_list), np.mean(sigma)))
            scores.append(score_avg)
            mean_loss.append(np.mean(loss_list))
            episodes.append(e)
            plot_reward(scores,mean_loss, episodes, "score_"+str(e))	
        if e % 10 == 0:
            agent.model.save("./working/model_ep" + str(e))
	    	
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

train()
