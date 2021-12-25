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

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

env = gym.make('gym-snake:snake-v0')

class DQN:
    def __init__(self):
        self.number_of_actions = 3
        self.replay_size = 10
        self.update_target_on_n = 4
        self.memory = []

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.learning_rate = 0.01

        self.target_model = self.create_model()
        self.model = self.create_model()
        # self.target_model = tf.keras.models.load_model('./saved_models/model_ecount_37.h5')
        # self.model = tf.keras.models.load_model('./saved_models/model_ecount_37.h5')
        self.max_steps_per_episode = 1000
        self.episode_count = 0

    def create_model(self):
        model = keras.models.Sequential()
        model.add(layers.Dense(16, input_shape=(9,), kernel_initializer='he_normal', activation='relu', name='input'))
        model.add(layers.Dense(8, kernel_initializer='he_normal', activation='relu', name='hidden1'))
        model.add(layers.Dense(units=self.number_of_actions, kernel_initializer='he_normal', activation='softmax', name='output'))
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(
            loss='huber',
            optimizer=opt,
            metrics=['mae'],
        )
        print(model.summary())
        return model

    # 0 forward, 1 left, 2 right
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, 2), False
        one_hot_state = []
        for s in state:
            temp_state = [0]*3
            temp_state[int(s)] = 1
            one_hot_state+=temp_state
        state = np.reshape(one_hot_state, (1,9))
        state = tf.reshape(state, (-1, state.shape[1]))
        a = self.model.predict(state)
        print(f"a: {a[0]}")
        print(f"a: {np.argmax(a[0, :])}")
        return np.argmax(a[0, :]), True

    def remember(self, state, action, reward, new_state, done):
        one_hot_state = []
        print(f"state: {state}")
        for s in state:
            temp_state = [0]*3
            temp_state[int(s)] = 1
            one_hot_state+=temp_state
        state = np.reshape(one_hot_state, (1,9))
        state = tf.reshape(state, (-1, state.shape[1]))
        print(f"new state: {new_state}")
        one_hot_state = []
        for s in new_state:
            temp_state = [0]*3
            temp_state[int(s)] = 1
            one_hot_state+=temp_state
        new_state = np.reshape(one_hot_state, (1,9))
        new_state = tf.reshape(new_state, (-1, state.shape[1]))


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
                # print("------------")
                # print(reward)
                # print(action)
                # print(target_f)
                target_f[0, action] = target
                # print(target_f)
                self.model.fit(batch_size=None, x=state, y=target_f, epochs=1, verbose=0)

agent = DQN()
score_history = []
render = True

try:     
    while True:
        cur_state = env.reset()
        episode_reward = 0
        done = False
        agent.epsilon *= agent.epsilon_decay
        agent.epsilon = max(agent.epsilon_min, agent.epsilon)
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
                break

        score_history.append(episode_reward)
        agent.episode_count += 1
        if env.snake_size == env.x_dim*env.y_dim:  # Condition to consider the task solved
            print(f"Solved at episode {agent.episode_count}!")
            agent.target_model.save('./saved_models/finished_model_ecount_'+str(agent.episode_count)+'.h5')
            break       
except:
    agent.target_model.save('./saved_models/model_ecount_'+str(agent.episode_count)+'.h5')
    # plot reward
    fig, ax = plt.subplots()
    ax.plot(range(0, agent.episode_count), score_history)
    ax.set(xlabel='episode', ylabel='end score')
    ax.grid()
    fig.savefig(f"{agent.episode_count}.png")
    plt.show()