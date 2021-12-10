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
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

        self.target_model = self.create_model()
        self.model = self.create_model()
        self.max_steps_per_episode = 1000
        self.episode_count = 0

    def create_model(self):
        model = keras.models.Sequential()
        model.add(layers.Dense(1, input_shape=(1, env.observation_space.shape[0]*env.observation_space.shape[1]*3+4), kernel_initializer='he_normal', activation='relu', name='input'))
        model.add(layers.Dense(units=64, kernel_initializer='he_normal', activation='relu', name='layer1'))
        model.add(layers.Dense(units=self.number_of_actions, kernel_initializer='he_normal', activation='softmax', name='output'))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            loss='huber',
            optimizer=opt,
            metrics=['mae'],
        )
        print(model.summary())
        return model

    # 0 forward, 1 left, 2 right
    def get_action(self, state, direction):
        if np.random.random() < self.epsilon:
            return random.randint(0, 2), False
        direction = self.one_hot_direction(direction)
        direction = direction.reshape(1, direction.shape[0])
        state = (np.arange(state.max()) == state[...,None]-1).astype(int)
        state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2])
        combined_state = np.append(direction, state, axis=1)
        combined_state = tf.reshape(combined_state, (-1, 1, combined_state.shape[1]))
        a = self.model.predict(combined_state)
        print(f"a: {a}")
        print(f"a: {np.argmax(a[0, :])}")
        return np.argmax(a[0, :]), True

    def one_hot_direction(self, direction):
        temp_dir = [0, 0, 0, 0]
        if direction==(0, 1): # up
            temp_dir[0] = 1
        elif direction==(-1, 0): # left
            temp_dir[1] = 1
        elif direction==(1, 0): # right
            temp_dir[2] = 1
        else: # down
            temp_dir[3] = 1
        return np.array(temp_dir)


    def remember(self, state, direction, action, reward, new_state, new_direction, done):
        direction = self.one_hot_direction(direction)
        direction = direction.reshape(1, direction.shape[0])
        state = (np.arange(state.max()) == state[...,None]-1).astype(int)
        state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2])
        combined_state = np.append(direction, state, axis=1)
        new_direction = self.one_hot_direction(new_direction)
        new_direction = new_direction.reshape(1, new_direction.shape[0])
        new_state = (np.arange(new_state.max()) == new_state[...,None]-1).astype(int)
        new_state = new_state.reshape(1, new_state.shape[0]*new_state.shape[1]*new_state.shape[2])
        combined_new_state = np.append(new_direction, new_state, axis=1)

        combined_state = tf.reshape(combined_state, (-1, 1, combined_state.shape[1]))
        combined_new_state = tf.reshape(combined_new_state, (-1, 1, combined_new_state.shape[1]))
        self.memory.append([combined_state, action, reward, combined_new_state, done])

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
                target_f[0, :, action] = [target]
                # print(target_f)
                self.model.fit(batch_size=None, x=state, y=target_f, epochs=1, verbose=0)

agent = DQN()
score_history = []
render = True

# try:     
while True:
    cur_state, cur_direction = env.reset()
    episode_reward = 0
    done = False
    agent.epsilon *= agent.epsilon_decay
    agent.epsilon = max(agent.epsilon_min, agent.epsilon)
    for timestep in range(1, agent.max_steps_per_episode):
        print("-------------")
        if render:
            env.render()
        action, greedy = agent.get_action(cur_state, cur_direction)

        # Apply the sampled action in our environment
        new_state, new_direction, reward, done, ___ = env.step(action)
        episode_reward+=reward

        agent.remember(cur_state, cur_direction, action, reward, new_state, new_direction, done)
        agent.replay()

        # update target model
        if timestep%agent.update_target_on_n==0:
            agent.target_model.set_weights(agent.model.get_weights())

        # Log details
        print(f"episode count: {agent.episode_count}, timestep: {timestep}, snake size: {env.snake_size}, episode reward: {episode_reward}")
        cur_state = new_state
        cur_direction = new_direction
        if done:
            break

    score_history.append(env.snake_size)
    agent.episode_count += 1
    if env.snake_size == env.x_dim*env.y_dim:  # Condition to consider the task solved
        print(f"Solved at episode {agent.episode_count}!")
        agent.target_model.save('./saved_models/finished_model_ecount_'+str(agent.episode_count)+'.h5')
        break       
# except:
#     agent.target_model.save('./saved_models/model_ecount_'+str(agent.episode_count)+'.h5')
#     # plot reward
#     fig, ax = plt.subplots()
#     ax.plot(range(0, agent.episode_count), score_history)
#     ax.set(xlabel='episode', ylabel='end score')
#     ax.grid()
#     fig.savefig(f"{agent.episode_count}.png")
#     plt.show()