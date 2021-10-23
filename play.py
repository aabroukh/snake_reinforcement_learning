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
        self.memory = []

        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.01
        self.tau = 0.125

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target = 1000
        self.max_steps_per_episode = 1000
        self.episode_count = 0

    def create_model(self):
        model = keras.models.Sequential()
        model.add(layers.InputLayer(batch_input_shape=(1, env.observation_space.shape[0], env.observation_space.shape[1])))
        model.add(layers.Dense(64, activation="relu", name='hidden1'))
        model.add(layers.Dense(64, activation='relu', name='hidden2'))
        model.add(layers.Dense(64, activation='relu', name='hidden3'))
        model.add(layers.Dense(self.number_of_actions, activation='softmax', name='output'))
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['mae'],
        )
        print(model.summary())
        return model

    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randint(0, 2), False
        a = self.model.predict(state)
        return np.argmax(a[0]), True

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) > 10:
            target = 0
            batch = random.sample(self.memory, 10)
            for state, action, reward, new_state, done in batch:
                if not done:
                    target = reward + self.gamma * np.amax(self.model.predict(new_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
        print("replay")

    def target_train(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

agent = DQN()
score_history = []
render = True

# try:     
while True:
    cur_state = env.reset()
    episode_reward = 0
    done = False
    for timestep in range(1, agent.max_steps_per_episode):
        if render:
            env.render()
        action, greedy = agent.get_action(cur_state)
        # Apply the sampled action in our environment
        new_state, reward, done, ___ = env.step(action)
        episode_reward+=reward

        agent.remember(cur_state, action, reward, new_state, done)
        agent.replay()

        # Log details
        print(f"episode count: {agent.episode_count}, timestep: {timestep}, greedy: {str(greedy)}, snake size: {env.snake_size}, episode reward: {episode_reward}")
        cur_state = new_state
        if done:
            break

    score_history.append(env.snake_size)
    agent.episode_count += 1
    if env.snake_size == env.x_dim*env.y_dim:  # Condition to consider the task solved
        print(f"Solved at episode {agent.episode_count}!")
        agent.target_model.save('./saved_models/finished_model_ecount_'+str(agent.episode_count)+'.h5')
        break       
# except:
agent.model.save('./saved_models/model_ecount_'+str(agent.episode_count)+'.h5')
agent.target_model.save('./saved_models/model_target_ecount_'+str(agent.episode_count)+'.h5')
# plot reward
fig, ax = plt.subplots()
ax.plot(range(0, agent.episode_count), score_history)
ax.set(xlabel='episode', ylabel='end score')
ax.grid()
fig.savefig(f"{agent.episode_count}.png")
plt.show()