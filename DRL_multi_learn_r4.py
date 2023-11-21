import math
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from collections import deque
import time
from DRL_env import DRLenv



class DRLmultiagent(object):
    def __init__(self, state_size, action_size, action_cand, pmax, noise):
        self.TTIs = 1000
        self.simul_rounds = 1

        self.EPSILON = 0.2

        self.initial_learning_rate = 5e-3
        self.learning_rate = self.initial_learning_rate
        self.lambda_lr = 1e-4  # decay rate for learning rate
        #self.learning_rate_decay = 1-math.pow(10,-4)

        self.gamma = 0.5

        self.pmax = pmax #38dbm

        self.state_size = state_size
        self.action_size = action_size
        self.action_cand = action_cand
        self.action_set = np.linspace(0, self.pmax, self.action_cand)


        #self.action = np.zeros(action_size)

        self.transmitters = 3
        self.users = 3

        self.env = DRLenv()
        self.A = self.env.tx_positions_gen()
        self.B = self.env.rx_positions_gen(self.A)

        self.noise = noise

        self.model = self.build_network

        self.replay_buffer = deque(maxlen=5000)
        self.update_rate = 100
        self.main_network = self.build_network()
        weight = self.main_network.get_weights()

        for i in range(1, self.transmitters + 1):
            setattr(self, f'target_network{i}', self.build_network())
            getattr(self, f'target_network{i}').set_weights(weight)

        self.loss = []

        self.temp_reward1 = 0

    def update_learning_rate(self):
        self.learning_rate *= (1 - self.lambda_lr)
        for i in range(1, self.transmitters + 1):
            getattr(self, f'target_network{i}').optimizer.lr.assign(self.learning_rate)
        self.main_network.optimizer.lr.assign(self.learning_rate)



    def build_network(self):
        model = Sequential()
        model.add(Dense(200, activation="tanh", input_shape=(self.state_size,)))
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(40, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def store_transistion(self, state, action, reward, next_state, done, agent):
        self.replay_buffer.append((state, action, reward, next_state, done, agent))

    def epsilon_greedy(self, agent, state, epsilon):
        if np.random.random() <= epsilon:
            action_temp = np.random.choice(len(self.action_set))
            action = self.action_set[int(action_temp)]
            print('EPS agent: ', agent, 'power: ', action)

        else:
            Q_values = getattr(self, f'target_network{agent + 1}').predict(state.reshape(1, -1))
            #Q_values = self.main_network.predict(state.reshape(1, -1))
            action_temp = np.argmax(Q_values[0])
            action = self.action_set[int(action_temp)]
            print('GRD agent: ', agent, 'power: ', action)

        return action

    def step(self, state, actions, TTI, max_TTI, channel_gain, next_channel_gain, agent):

        if TTI >= max_TTI:
            done = True
        else:
            done = False

        old_state = np.copy(state)
        next_state = np.zeros([self.state_size])

        reward = 0
        self.temp_reward1 = 0
        temp_reward2 = 0

        action_of_agent = actions[agent]
        inter = 0

        direct_signal = channel_gain[agent, agent] * action_of_agent

        for j in range(self.transmitters):
            if j == agent:
                inter += 0
            else:
                action_of_interferer = actions[j]
                gain_temp_interferer = channel_gain[j, agent]
                inter_of_interferer = gain_temp_interferer * action_of_interferer
                inter += inter_of_interferer

        self.temp_reward1 = math.log2(1 + direct_signal / (inter + self.noise))

        for j in range(self.users):
            inter_of_interfered = 0
            inter_of_interfered_without_agent = 0
            if j == agent:
                temp_reward2 += 0
            else:

                for k in range(self.transmitters):
                    if k == j:
                        inter_of_interfered += 0
                        inter_of_interfered_without_agent += 0
                    else:
                        if k != agent:
                            action_to_interfered = actions[k]
                            gain_temp_interferer = channel_gain[k, j]
                            inter_to_interfered = gain_temp_interferer * action_to_interfered
                            inter_of_interfered += inter_to_interfered
                            inter_of_interfered_without_agent += inter_to_interfered

                        else:
                            action_to_interfered = actions[k]
                            gain_temp_interferer = channel_gain[k, j]
                            inter_to_interfered = gain_temp_interferer * action_to_interfered
                            inter_of_interfered += inter_to_interfered
                            inter_of_interfered_without_agent += 0

                rate_with_agent = math.log2(1 + (channel_gain[j, j] * actions[j]) / (inter_of_interfered + self.noise))
                rate_without_agent = math.log2(
                    1 + (channel_gain[j, j] * actions[j]) / (inter_of_interfered_without_agent + self.noise))
                temp_reward2 += (rate_without_agent - rate_with_agent)

        reward = self.temp_reward1 - temp_reward2

        next_state[0] = actions[agent]
        next_state[1] = self.temp_reward1
        next_state[2] = next_channel_gain[agent, agent]
        next_state[3] = channel_gain[agent, agent]

        new_agent_inter = 0
        for j in range(self.transmitters):
            if j == agent:
                new_agent_inter += 0
            else:
                action_of_interferer = actions[j]
                gain_temp_interferer = next_channel_gain[j, agent]
                inter_of_interferer = gain_temp_interferer * action_of_interferer
                new_agent_inter += inter_of_interferer
        next_state[4] = new_agent_inter
        next_state[5] = old_state[4]

        for j in range(self.transmitters):
            if j != agent:
                next_state[4*j+6] = next_channel_gain[j, agent] * actions[j]

                direct_signal_tmp = channel_gain[j, j] * actions[j]

                inter_tmp = 0

                for k in range(self.transmitters):
                    if k == j:
                        inter_tmp += 0
                    else:
                        action_of_interferer_tmp = actions[k]
                        gain_temp_interferer_tmp = channel_gain[k, j]
                        inter_of_interferer_tmp = gain_temp_interferer_tmp * action_of_interferer_tmp
                        inter_tmp += inter_of_interferer_tmp

                next_state[4*j + 7] = math.log2(1 + direct_signal_tmp / (inter_tmp + self.noise))

                next_state[4*j + 8] = old_state[4*j+6]

                next_state[4 * j + 9] = old_state[4 * j + 7]

            else:
                next_state[4 * j + 6] = 0
                next_state[4 * j + 7] = 0
                next_state[4 * j + 8] = 0
                next_state[4 * j + 9] = 0

        for k in range(self.transmitters):
            if k != agent:
                next_state[3*k+6+4*self.transmitters] = channel_gain[k, k]

                direct_signal_tmp = channel_gain[k, k] * actions[k]
                inter_tmp = 0

                for m in range(self.transmitters):
                    if m == k:
                        inter_tmp += 0
                    else:
                        action_of_interferer_tmp = actions[m]
                        gain_temp_interferer_tmp = channel_gain[m, k]
                        inter_of_interferer_tmp = gain_temp_interferer_tmp * action_of_interferer_tmp
                        inter_tmp += inter_of_interferer_tmp

                next_state[3 * k + 7 + 4*self.transmitters] = math.log2(1 + direct_signal_tmp / (inter_tmp + self.noise))

                next_state[3 * k + 8 + 4 * self.transmitters] = (channel_gain[agent, k] * actions[agent]) / (inter_tmp + self.noise)

            else:
                next_state[3 * k + 6 + 4 * self.transmitters] = 0
                next_state[3 * k + 7 + 4 * self.transmitters] = 0
                next_state[3 * k + 8 + 4 * self.transmitters] = 0


        info = {}

        return next_state, reward, done, info




    def train(self, batch_size):

        self.update_learning_rate()

        # compute the Q value using the target network
        minibatch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, next_state, done, agent in minibatch:
            target_Q = reward
            if not done:
                max_future_q = np.amax(
                    getattr(self, f'target_network{agent + 1}').predict(next_state.reshape(1, -1))[0])
                target_Q += self.gamma * max_future_q

            current_qs = self.main_network.predict(state.reshape(1, -1))
            action_index = np.where(self.action_set == action)[0][0]
            current_qs[0][action_index] = target_Q

            # train the main network
            result = self.main_network.fit(state.reshape(1, -1), current_qs, epochs=1, verbose=1)

            self.loss.append(result.history['loss'])




    def update_target_network(self):
        weight = self.main_network.get_weights()
        for i in range(1, self.transmitters + 1):
            getattr(self, f'target_network{i}').set_weights(weight)
        return 0


