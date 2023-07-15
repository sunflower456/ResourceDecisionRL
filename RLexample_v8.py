# DQN + LSTM
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
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import random
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.fc1 = LSTM(64,return_sequences=True)
        self.fc2 = LSTM(64)
        self.fc3 = Dense(32, activation='relu')
        self.fc_out = Dense(action_size,
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.fc_out(x)
        return q

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

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
       
        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = DQN(self.action_size, self.state_size)
        # self.model.load_weights("./working/model_ep2000")
        self.target_model = DQN(self.action_size, self.state_size)
        self.optimizer = Adam(lr=self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def reset(self):
        self.num_up = 0
        self.num_down = 0
        self.num_stay = 0
        self.request = 3
        return 
        
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decide_action_test(self, state):
        q_value = self.model(state)
        action = np.argmax(q_value[0])
        return action, sigmoid(q_value[0][action])

    def decide_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), 0.5
        else:
            q_value = self.model(state)
            action = np.argmax(q_value[0])
            # print("state: ", state, " | action : ", action, " | q_value[0] : ", q_value[0])
            return action, sigmoid(q_value[0][action])
        
    def decide_scaling_unit(self, request, resourceList, action, confidence):
        scaling_unit = 0.
        if action == 1: # upscaling
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

        elif action == 2: #downscaling
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
        if action == 1:
            # up할 단위를 판단
            self.num_scaling += 1 # scaling 횟수 증가
            self.num_up += 1 # upscaling 횟수 증가
            if (usage) < self.upper_bound:
                reward = -1
            else:
                self.request = curr_request + scaling_unit
                reward = 1
            print("[UP] reward : ", reward, " || request : ", self.request, " || resource : ", resource, " || usage : ", usage, " || scale : ", scaling_unit)
        
                        
        # downscaling
        elif action == 2:
            # donw할 단위를 판단
            self.num_scaling += 1
            self.num_down = self.num_down + 1

            if (usage) >= self.lower_bound:
                reward = -1
            else:
                self.request = curr_request - scaling_unit
                reward = 1
            print("[DOWN] reward : ", reward, " || request : ", self.request, " || resource : ", resource, " || usage : ", usage, " || scale : ", scaling_unit)
            
        # 관망
        else:
            self.num_scaling += 1
            self.num_stay += 1  # 관망 횟수 증가
            if ((usage) >= self.lower_bound) & ((usage) < self.upper_bound):
                reward = 1 # 1 -> 1.5
            else:
                reward = -1
            print("[STAY] reward : ", reward, " || request : ", self.request, " || resource : ", resource, " || usage : ", usage)

        
        return reward, self.request
			
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 학습 파라메터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss
            

# returns the vector containing stock data from a fixed file
def getResourceDataVec(key):
	vec = []
	lines = open("./input/"+key+".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(round(float(line.split(",")[3]) / 10000000, 2))

	return vec


# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        xs.append(x)
    # random.shuffle(xs)    
    return np.array(xs)

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
        state = np.reshape(state, [1, 1, window_size])
        # state = np.reshape(state, [1, window_size])
        agent.reset()
        loss = 0
        for t in range(l-window_size):
            action, confidence = agent.decide_action(state)
                    
            reward, request = agent.act(action, data[t], resource, l, confidence)
            
            next_state, next_resource = getState(data, t+1, window_size+1, request) 
            next_state = np.reshape(next_state, [1, 1, window_size])
            
            done = True if reward < 0 else False
            temp_reward = 0.1 if reward > 0 else -1
            

            agent.append_sample(state, action, temp_reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                loss = agent.train_model()
                if loss > 1:
                    print("[",t,"] loss : ", loss)

            mean_loss.append(loss)
            total_reward += reward
            state = next_state
            resource = next_resource
            
            if t == l-1-window_size:
                agent.update_target_model()
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
    
    data, window_size, episode_count = 'kubernetes_pod_container_zipkin_test', 12, 10

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
