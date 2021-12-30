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
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001

        self.target_model = self.create_model()
        self.model = self.create_model()
        # self.target_model = tf.keras.models.load_model('./saved_models/model_ecount_105.h5')
        # self.model = tf.keras.models.load_model('./saved_models/model_ecount_105.h5')
        self.max_steps_per_episode = 1000
        self.episode_count = 0

    def create_model(self):
        model = keras.models.Sequential()
        model.add(layers.Conv2D(
            128, 
            kernel_size=(3, 3), 
            kernel_initializer='he_normal', 
            strides=(1, 1),
            activation='relu', 
            padding='same', 
            input_shape=(15, 15, 1), 
            name='input'))
        model.add(layers.Conv2D(
            128, 
            kernel_size=(3, 3), 
            kernel_initializer='he_normal',
            strides=(1, 1), 
            activation='relu', 
            padding='same', 
            name='hidden1'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_initializer='he_normal', activation='relu', name='hidden2'))
        model.add(layers.Dense(512, kernel_initializer='he_normal', activation='relu', name='hidden3'))
        model.add(layers.Dense(units=self.number_of_actions, kernel_initializer='he_normal', activation='softmax', name='output'))
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss='huber',
            optimizer=opt
        )
        print(model.summary())
        return model
    
    def one_hot_encode(self, state, optimal=False):
        # if optimal:
        #     print(f"after: {state}")
        state = tf.reshape(state, (-1, 15, 15, 1))
        return state

    # 0 left, 1 forward, 2 right
    def get_action(self, state):
        # print(f"state: {state}")
        if np.random.random() < self.epsilon:
            return random.randint(0, 2), False
        print("optimal")
        state = self.one_hot_encode(state, True)
        a = self.model.predict(state)
        print(f"a: {a[0]}")
        print(f"a: {np.argmax(a[0, :])}")
        return np.argmax(a[0, :]), True

    def remember(self, state, action, reward, new_state, done):
        state = self.one_hot_encode(state)
        new_state = self.one_hot_encode(new_state)
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
render = False

try:     
    while True:
        cur_state = env.reset()
        episode_reward = 0
        done = False
        agent.epsilon *= agent.epsilon_decay
        # agent.epsilon = max(agent.epsilon_min, agent.epsilon)
        for timestep in range(1, agent.max_steps_per_episode):
            print("-------------")
            if render:
                env.render()
            action, greedy = agent.get_action(cur_state)

            # Apply the sampled action in our environment
            new_state, reward, done, ___ = env.step(action)
            episode_reward+=reward
            agent.remember(cur_state, action, reward, new_state, done)
            agent.replay()

            # update target model
            if timestep%agent.update_target_on_n==0:
                agent.target_model.set_weights(agent.model.get_weights())

            # Log details
            print(f"episode count: {agent.episode_count}, timestep: {timestep}, snake size: {env.snake_size}, episode reward: {episode_reward}")
            cur_state = new_state
            if done:
                print("died")
                break

        score_history.append(env.snake_size-3)
        agent.episode_count += 1
        if env.snake_size == env.x_dim*env.y_dim:  # Condition to consider the task solved
            print(f"Solved at episode {agent.episode_count}!")
            agent.target_model.save('./saved_models/finished_model_ecount_'+str(agent.episode_count)+'.h5')
            break       
except:
    agent.target_model.save('./saved_models/model_ecount_'+str(agent.episode_count)+'.h5')
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