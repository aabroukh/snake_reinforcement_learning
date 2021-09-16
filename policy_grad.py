import gym
import gym_snake



env = gym.make('gym-snake:snake-v0')
for episode in range(10000):
    observation = env.reset()
    ended = False
    while not ended:
        env.render()
        action = env.action_space.sample()
        observation, reward, ended, info = env.step(action)
        print(action, reward)