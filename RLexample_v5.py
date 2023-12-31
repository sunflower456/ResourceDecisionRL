# A2C
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


# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.actor_fc = Dense(24, activation='tanh')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1,
                                kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value

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

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
       
        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.action_size)
        self.model.load_weights("./model/a2c_kibana")
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        # self.optimizer = Adam(lr=self.learning_rate, clipnorm=5.0)


    def reset(self):
        self.num_up = 0
        self.num_down = 0
        self.num_stay = 0
        self.request = 3
        return 
        
        

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy, _ = self.model.predict(state)
        policy = np.array(policy[0])
        # print("state: ", state, " | policy : ", policy, " | action : ",np.random.choice(self.action_size, 1, p=policy)[0])
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
			
    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy, value = self.model(state)
            _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)


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
            
            loss = agent.train_model(state, action, temp_reward, next_state, done)

            if loss > 1:
                print("[",t,"] : loss : ", loss)

            mean_loss.append(loss)
            total_reward += reward
            state = next_state
            resource = next_resource
            
            if t == l-1-window_size:
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
    
    data, window_size, episode_count = 'kubernetes_pod_container_zipkin_test', 6, 10

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
        agent.reset()
        for t in range(l-window_size):
            action, confidence = agent.decide_action_test(state)
                    
                    
            reward, request = agent.act(action, data[t], resource, l, confidence)
            
            next_state, next_resource = getState(data, t+1, window_size+1, request)    
            
            total_reward += reward
            state = next_state
            resource = next_resource

            if t == l-1-window_size:
                total_rewards.append(total_reward)
                episodes.append(e)
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Total rewards: ", total_reward, " || last request :", agent.request, " || num_up : ", agent.num_up, " || num_down: ", agent.num_down, " || num_stay : ", agent.num_stay)
                print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


        if (e % 10 == 0) & (e != 0):
            plt.plot(episodes, total_rewards, 'b')
            plt.xlabel("episode")
            plt.title('score')
            plt.ylabel('average score')
            plt.savefig("score_test_"+str(e)+".png")

train()
