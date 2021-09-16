import gym
import gym_snake

env = gym.make('gym-snake:snake-v0')

for episode in range(10):
    observation = env.reset()
    done = False
    
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
env.close()