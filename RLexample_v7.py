# REINFORCE
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
print(os.listdir("./input"))
print(os.listdir("./working"))

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import random
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # stay, watching, up, down
        self.memory = deque(maxlen=1000)
        self.request = 3
        self.num_up = 0  # upscaling 횟수
        self.num_down = 0  # downscaling 횟수
        self.num_stay = 0  # stay 횟수
        self.num_scaling = 1 
        self.upper_bound = 1.0
        self.lower_bound = 0.5

        self.batch_size = 36
        self.train_start = 1000
        self.is_eval = is_eval
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE(self.action_size)
        self.model.load_weights("./model/reinforce_kibana")
        # self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    def reset(self):
        self.num_up = 0
        self.num_down = 0
        self.num_stay = 0
        self.request = 3
        return 
        
    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

        
    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)


    # 정책신경망으로 행동 선택
    def get_action_test(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0],0.5

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model(state)[0]
        policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0], 0.5

    def decide_scaling_unit(self, request, resourceList, action, confidence):
        scaling_unit = 0.
        if action == 0: # upscaling
            tempList = []
            tempList2 = []
            for i in range(len(resourceList)-1):
                tag = True if resourceList[i]/request >= self.upper_bound else False
                if tag == False:
                    break
                else:
                    tempList.append(abs(request - resourceList[i]))
            for i in range(len(resourceList)-1):
                tempList2.append(abs(request-resourceList[i]))
            if not tempList:
                scaling_unit = min(tempList2)
            else:
                scaling_unit = np.mean(tempList)

        elif action == 1: #downscaling
            tempList = []
            tempList2 = []
            for i in range(len(resourceList)-1):
                tag = True if resourceList[i]/request < self.lower_bound else False
                if tag == False:
                    break
                else:
                    tempList.append(abs(request - resourceList[i]))
            for i in range(len(resourceList)-1):
                tempList2.append(abs(request-resourceList[i]))
            if not tempList:
                scaling_unit = min(tempList2)
            else:
                scaling_unit = np.mean(tempList)
        # print("request: ", request, " | scaling_unit : ", scaling_unit, " | resourceList: ", resourceList)
        return round(float(scaling_unit),3)


    def act(self, action, resource, resourceList, l, confidence):
        curr_request = self.request
        scaling_unit = self.decide_scaling_unit(curr_request, resourceList[0], action, confidence)
        usage = resource/curr_request
        reward = 0
        # upscaling
        if action == 0:
            # up할 단위를 판단
            self.num_scaling += 1 # scaling 횟수 증가
            self.num_up += 1 # upscaling 횟수 증가
            if (usage) < self.upper_bound:
                reward = -1
            else:
                self.request = curr_request + scaling_unit
                reward = 1
            # print("[UP] reward : ", reward, " || request : ", self.request, " || resource : ", resource, " || usage : ", usage, " || scale : ", scaling_unit)
        
                        
        # downscaling
        elif action == 1:
            # donw할 단위를 판단
            self.num_scaling += 1
            self.num_down = self.num_down + 1

            if (usage) >= self.lower_bound:
                reward = -1
            else:
                self.request = curr_request - scaling_unit
                reward = 1
            # print("[DOWN] reward : ", reward, " || request : ", self.request, " || resource : ", resource, " || usage : ", usage, " || scale : ", scaling_unit)
            
        # 관망
        else:
            self.num_scaling += 1
            self.num_stay += 1  # 관망 횟수 증가
            if ((usage) >= self.lower_bound) & ((usage) < self.upper_bound):
                reward = 1 # 1 -> 1.5
            else:
                reward = -1
            # print("[STAY] reward : ", reward, " || request : ", self.request, " || resource : ", resource, " || usage : ", usage)

        
        return reward, self.request
			

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)

# returns the vector containing stock data from a fixed file
def getResourceDataVec(key):
	vec = []
	lines = open("./input/"+key+".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(round(float(line.split(",")[3]) / 10000000, 2))

	return vec


# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


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


def getState(data, t, n, request):
    block = data[t:t + n] if t < len(data)-n-1 else data[len(data)-n-1:len(data)]
    if t == 0:
        request = 1
    state = []
    resource = []
    for i in range(n - 1):
        state.append(sigmoid(block[i]/request))
        resource.append(block[i])
    
    # scaler = RobustScaler()
    # state = np.reshape(state, [-1,1])
    # state = scaler.fit_transform(state)    
    return np.array([state]), np.array([resource])


def train():
    
    data, window_size, episode_count = 'kubernetes_pod_container_kibana_train', 4, 5000 # 25

    agent = Agent(window_size)
    data = getResourceDataVec(data)
    l = len(data) - 1
    episodes = []
    total_rewards = []
    total_losses = []
    for e in range(episode_count + 1):
        total_reward = 0
        mean_loss = []
        print("Episode " + str(e) + "/" + str(episode_count))
        request = agent.request
        done = False
        state, resource = getState(data, 0, window_size+1, request)
        state = np.reshape(state, [1, window_size])
        agent.reset()
        loss = 0
        for t in range(l-window_size):
            action, confidence = agent.get_action(state)
                    
            reward, request = agent.act(action, data[t], resource, l, confidence)
            
            next_state, next_resource = getState(data, t+1, window_size+1, request) 
            next_state = np.reshape(next_state, [1, window_size])   
            
            done = True if reward < 0 else False
            temp_reward = 0.1 if reward > 0 else -1
            
            agent.append_sample(state, action, temp_reward)
            
            total_reward += reward
            state = next_state
            resource = next_resource
            
            if t == l-1-window_size:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                mean_loss.append(entropy)
                total_rewards.append(total_reward)
                total_losses.append(np.mean(mean_loss))
                episodes.append(e)
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Total rewards: ", total_reward, " || Mean loss: ", np.mean(mean_loss), " || last request :", agent.request, " || num_up : ", agent.num_up, " || num_down: ", agent.num_down, " || num_stay : ", agent.num_stay)
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


        if (e % 100 == 0) & (e != 0):
            agent.model.save("./working/model_ep" + str(e))
            plot_reward(total_rewards, total_losses, episodes, "score_"+str(e))

def test():
    
    data, window_size, episode_count = 'kubernetes_pod_container_kibana_test', 4, 10

    agent = Agent(window_size)
    data = getResourceDataVec(data)
    l = len(data) - 1
    episodes = []
    total_rewards = []
    for e in range(episode_count + 1):
        total_reward = 0
        mean_loss = []
        print("Episode " + str(e) + "/" + str(episode_count))
        request = agent.request
        done = False
        state, resource = getState(data, 0, window_size+1, request)
        state = np.reshape(state, [1, window_size])
        agent.reset()
        for t in range(l-window_size):
            action, confidence = agent.get_action_test(state)
                    
            reward, request = agent.act(action, data[t], resource, l, confidence)
            
            next_state, next_resource = getState(data, t+1, window_size+1, request) 
            next_state = np.reshape(next_state, [1, window_size])   
            
            done = True if reward < 0 else False
            temp_reward = 0.1 if reward > 0 else -1

            total_reward += reward
            state = next_state
            resource = next_resource
            
            if t == l-1-window_size:
                total_rewards.append(total_reward)
                episodes.append(e)
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Total rewards: ", total_reward, " || last request :", agent.request, " || num_up : ", agent.num_up, " || num_down: ", agent.num_down, " || num_stay : ", agent.num_stay)
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

test()
