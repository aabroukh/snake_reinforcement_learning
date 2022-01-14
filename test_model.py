import gym
import gym_snake
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.keras import backend as K
from tensorflow import keras
from collections import deque 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Sequential
from tensorflow.python.ops.gen_array_ops import one_hot

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

env = gym.make('gym-snake:snake-v0')

class DQN:
    def __init__(self):
        self.number_of_actions = 3
        self.replay_size = 10
        self.update_target_on_n = 5
        self.memory = []

        self.gamma = 0.95
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.992
        self.learning_rate = 0.00025

        # self.target_model = self.create_model()
        # self.model = self.create_model()
        self.target_model = tf.keras.models.load_model('./saved_models/model_ecount_1000.h5')
        self.model = tf.keras.models.load_model('./saved_models/model_ecount_1000.h5')
        self.max_steps_per_episode = 1000
        self.episode_count = 0

    def create_model(self):
        model = keras.models.Sequential()
        model.add(layers.Dense(128, input_shape=(10,), kernel_initializer='he_normal', activation='relu', name='input'))
        model.add(layers.Dense(128, kernel_initializer='he_normal', activation='relu', name='hidden1'))
        model.add(layers.Dense(128, kernel_initializer='he_normal', activation='relu', name='hidden2'))
        model.add(layers.Dense(units=self.number_of_actions, kernel_initializer='he_normal', activation='softmax', name='output'))
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss='huber',
            optimizer=opt
        )
        print(model.summary())
        return model
    
    def one_hot_encode(self, state, food_dir, optimal=False):
        one_hot_state = []
        for s in state:
            temp_state = [0]*2
            if s:
                temp_state[int(s)-1] = 1
            one_hot_state+=temp_state
        one_hot_state+=food_dir
        state = np.reshape(one_hot_state, (1,10))
        if optimal:
            print(f"after: {state}")
        state = tf.reshape(state, (-1, state.shape[1]))
        return state

    # 0 left, 1 forward, 2 right
    def get_action(self, state, food_dir):
        print(f"state: {state}")
        print(f"food_dir: {food_dir}")
        if np.random.random() < self.epsilon:
            return random.randint(0, 2), False
        print("optimal")
        state = self.one_hot_encode(state, food_dir, True)
        a = self.model.predict(state)
        print(f"a: {a[0]}")
        print(f"a: {np.argmax(a[0, :])}")
        return np.argmax(a[0, :]), True

    def remember(self, state, food_dir_cur, action, reward, new_state, food_dir_new, done):
        state = self.one_hot_encode(state, food_dir_cur)
        new_state = self.one_hot_encode(new_state, food_dir_new)
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) > self.replay_size:
            target = 0
            batch = random.sample(self.memory, self.replay_size)
            for state, action, reward, new_state, done in batch:
                if done:
                    target = reward
                else: 
                    target = reward + self.gamma * np.amax(self.target_model.predict(new_state))
                target_f = self.model.predict(state)
                target_f[0, action] = target
                self.model.fit(batch_size=None, x=state, y=target_f, epochs=1, verbose=0)

agent = DQN()
score_history = []
render = True

try:     
    while True:
        cur_state, food_dir = env.reset()
        done = False
        # agent.epsilon = max(agent.epsilon_min, agent.epsilon)
        for timestep in range(1, agent.max_steps_per_episode):
            if render:
                env.render()
            action, greedy = agent.get_action(cur_state, food_dir)

            # Apply the sampled action in our environment
            new_state, new_food_dir, reward, done, ___ = env.step(action)

            if done:
                print("died")
                break

        score_history.append(env.snake_size-3)
        agent.episode_count += 1
        if agent.episode_count==100: 
            # print(agent.target_model.layers[0].get_weights())
            # print(agent.target_model.layers[1].get_weights())
            # plot reward
            fig, ax = plt.subplots()
            ax.plot(range(0, agent.episode_count), score_history)
            ax.set(xlabel='episode', ylabel='end score')
            ax.grid()
            fig.savefig(f"./saved_figures/{agent.episode_count}.png")
            plt.show()
            env.close()
            break       
except:
    # print(agent.target_model.layers[0].get_weights())
    # print(agent.target_model.layers[1].get_weights())
    # plot reward
    fig, ax = plt.subplots()
    ax.plot(range(0, agent.episode_count), score_history)
    ax.set(xlabel='episode', ylabel='end score')
    ax.grid()
    fig.savefig(f"./saved_figures/{agent.episode_count}.png")
    plt.show()
    env.close()