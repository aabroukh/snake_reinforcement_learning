from . import envs
from gym.envs.registration import register

__version__ = "0.0.1"

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv'
)
