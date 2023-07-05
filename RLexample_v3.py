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
from sklearn.preprocessing import MinMaxScaler

class ContinuousA2CEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, request):
        self.request = request
        self.max_action =  0.2
        self.observation = []

        self.action_space = spaces.Box(
            low=-self.max_action,
            high=self.max_action,
            shape=(1,)
        )
        
        self.state = None
        self.seq = []
        self.seed()
	
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, data, n):
        seq = create_sequences(data, n)
        self.seq = seq.copy()
        temp = []
        for i in range(len(seq[0])):
            temp.append(seq[0][i] - self.request)
        self.state = temp.copy()
        # self.state = self.seq[0]
        return np.array(self.state, dtype=np.float32)

    def step(self, action, dataLen, request, t, n):
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        reward = 0
        done = False
        currState = []
        nextState = []

        if t + n >= dataLen:
            currState = self.seq[dataLen-n-1].copy()
        else:
            currState = self.seq[t].copy()

        nextState = currState.copy()    
       
        for j in range(len(nextState)):
            currState[j] = currState[j] - request 
            nextState[j] = nextState[j] - (request + action[0]) 
            
        if currState[0] > 0 : # need to up
            if action < 0 : # but down
                done = True
                # print("1 | action: {:3f} | reward: {:.3f} | currState[0]: {:.3f} | nextState[0]: {:.3f}".format(action[0], reward, currState[0][0], nextState[0][0]))
            else: #good! up
                if currState[0] <= action:
                    reward = 1
                else:
                    reward = 0.5
        else: #need to down
            if action > 0: #but up
                done = True
                # print("2 | action: {:3f} | reward: {:.3f} | currState[0]: {:.3f}| nextState[0]: {:.3f}".format(action[0], reward, currState[0][0], nextState[0][0]))
            else: # good! downv
                if (currState[1] < 0) & (currState[2] < 0):
                    reward = 1 
                else:
                    reward = 0.5
        if (request + action[0] < 0) :
            done = True
            reward = 0
            # print("3 | action: {:3f} | reward: {:.3f} | currState[0]: {:.3f} | nextState[0]: {:.3f}".format(action[0], reward, currState[0][0], nextState[0][0]))

        
        for j in range(len(nextState)):
            nextState[j] = nextState[j] + request
            if nextState[j] < -1:
                nextState[j] = -1
            elif nextState[j] > 1:
                nextState[j] = 1
            
        # print("curr: ", currState)
        # print("next: ", nextState)

        return np.array(nextState, dtype=np.float32), np.array(currState, dtype=np.float32), reward, request, done
	
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
    def __init__(self, action_size, max_action, request):
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size
        self.max_action = max_action
        self.request = request
	
        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99 # 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = ContinuousA2C(self.action_size)

        self.model.load_weights("./working/model_ep100")

        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=1.0)


    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0] 
        action = np.clip(action, -self.max_action, self.max_action)
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

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        xs.append(x)
    np.random.shuffle(xs)
    return xs


def train():
    name, window_size, episode_count = 'kubernetes_pod_container_behavior_20230531', 3, 5000 

    data = getStockDataVec(name)
    # request = 2.5

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = np.reshape(data, [-1, 1])
    # request = np.reshape(request, [-1, 1])
    data = scaler.fit_transform(data)
    # request = scaler.transform(request)
    # request = request[0][0]
    request = 0.78
    env = ContinuousA2CEnv(request)
    state_size = window_size
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0] 
    agent = ContinuousA2CAgent(action_size, max_action, request)

    print("request: ", request, " | action_size: ", action_size, " | max_action: ", max_action)
    l = len(data) - 1
    scores, mean_loss, episodes = [], [], []
    score_avg = 0
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        loss_list, sigma_list = [], []
        done = False
        score = 0
        loss = 0
        sigma = 0
        state = env.reset(data, window_size)
        state = np.reshape(state, [1, state_size])
        for t in range(l):
            action = agent.get_action(state)
            next_state, state, reward, request, done = env.step(action, l, request, t, window_size)
            state = np.reshape(state, [1, state_size])
            next_state = np.reshape(next_state, [1, state_size])
            score += reward
            reward = 0.1 if not done and t != l - 1 else -1
            loss, sigma = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            sigma_list.append(sigma)
            state = next_state
        score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
        print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f} | sigma: {:.3f}".format(e, score_avg, np.mean(loss_list), np.mean(sigma)))
        print("==============================done==============================")
        scores.append(score_avg)
        mean_loss.append(np.mean(loss_list))
        episodes.append(e)
        # plot_reward(scores,mean_loss, episodes, "score_"+str(e))	
        if e % 100 == 0:
            agent.model.save("./working/model_ep" + str(e))
            plot_reward(scores,mean_loss, episodes, "score_"+str(e))
		
        
    return 
			
def test():
    name, window_size, episode_count = 'kubernetes_pod_container_portal_20230624', 3, 100 

    data = getStockDataVec(name)
    # request = 2.5

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = np.reshape(data, [-1, 1])
    # request = np.reshape(request, [-1, 1])
    data = scaler.fit_transform(data)
    # request = scaler.transform(request)
    # request = request[0][0]
    request = 0.78
    env = ContinuousA2CEnv(request)
    state_size = window_size
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0] 
    agent = ContinuousA2CAgent(action_size, max_action, request)


    print("request: ", request)
    l = len(data) - 1
    scores, mean_loss, episodes = [], [], []
    score_avg = 0
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        loss_list, sigma_list = [], []
        done = False
        score = 0
        loss = 0
        sigma = 0
        state = env.reset(data, window_size)
        state = np.reshape(state, [1, state_size])
        for t in range(l):
            action = agent.get_action(state)
            print("state: ", state)
            print("action: ",action)
            next_state, state, reward, request, done = env.step(action, l, request, t, window_size)
            state = np.reshape(state, [1, state_size])
            next_state = np.reshape(next_state, [1, state_size])
            score += reward
            state = next_state
        print("episode: {:3d} | score avg: {:3.2f} ".format(e, score))
    return   

test()
