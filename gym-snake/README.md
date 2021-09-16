# Gym Snake

This repository contains an implementation of a classic game of 
Snake as a [gym](https://github.com/openai/gym/tree/master/gym/envs) environment. 
Implementation of this environment and RL algorithms to solve it was one
of the warmups in OpenAI's [Requests for Research 2.0](https://blog.openai.com/requests-for-research-2/).

## Installation
The environment doesn't use any dependencies other than the ones 
used by gym. So if you've managed to install gym. You should not
have trouble installing this environment.

To install the environment open the terminal and enter the following
commands:
1. `git clone https://github.com/AleksaC/gym-snake.git`
2. `cd gym-snake/gym-snake`
3. `python -m pip install .`

If you want to modify the environment you should use `python -m pip install -e .`
This will install the environment in 
[editable mode](https://pip.pypa.io/en/latest/reference/pip_install/?highlight=editable#editable-installs)
so any changes made to the code will be immediately reflected.

## Example
The following snippet runs the environment 10 times sampling random 
actions and renders the results on the screen.
```python
import gym
import gym_snake

env = gym.make('snake-v0')

for episode in range(10):
    observation = env.reset()
    done = False
    
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
env.close()
```
