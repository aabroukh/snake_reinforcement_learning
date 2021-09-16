from collections import deque
import time

import gym
import numpy as np

from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

import random


class SnakeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": "35"
    }
    # (1, 0) = up, (-1, 0) = down, (0, -1) = left, (0, 1) = right
    # 0 = forward, 1 = right, 2 = left
    def __init__(self, height=33, width=33, scaling_factor=6,
                 starting_position=(0, 0), snake_size=3, direction=(0, 0),
                 time_penalty=-0.01, food_reward=1, loss_penalty=-1, win_reward=10):
        self.action_space = spaces.Discrete(3)
        self.ACTIONS = ["STRAIGHT", "LEFT", "RIGHT"]
        self.observation_space = spaces.Box(0, 2, (height + 2, width + 2), dtype="uint8")
        self.viewer = None
        self.seed()

        # rewards and penalties
        self.time_penalty = time_penalty
        self.food_reward = food_reward
        self.loss_penalty = loss_penalty
        self.win_reward = win_reward
        if loss_penalty > 0 or time_penalty > 0:
            logger.warn("Values of penalties should not be positive.")

        # initialize size and position properties
        self.height = height
        self.width = width

        # randomize initial direction and position
        dir_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        direction = dir_list[random.randint(0, 3)]
        if direction == (1, 0): # up
            starting_position = [random.randint(snake_size+1, height), random.randint(1, width)]
        elif direction == (-1, 0): # down
            starting_position = [random.randint(1, height-snake_size), random.randint(1, width)]
        elif direction == (0, 1): # right
            starting_position = [random.randint(1, height), random.randint(snake_size+1, width)]
        elif direction == (0, -1): # left
            starting_position = [random.randint(1, height), random.randint(1, width-snake_size)]
        print(direction, starting_position)

        self.scaling_factor = scaling_factor
        self.initial_size = snake_size
        self.snake_size = snake_size
        self.max_size = height * width
        self.state = np.zeros((height + 2, width + 2), dtype="uint8")
        self.game_over = False

        # set bounds of the environment
        self.state[:, 0] = self.state[:, -1] = 1
        self.state[0, :] = self.state[-1, :] = 1

        # initialize snake properties
        self.direction = direction
        self.snake = deque()
        # initialize position of the snake
        self._init_field(starting_position, snake_size)

        # place food on the field
        self.food = self._generate_food()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_field(self, starting_position, snake_size):
        y, x = starting_position
        for i in range(snake_size):
            self.state[y][x] = 1
            if self.direction == (0, 1):
                x-=1
                self.snake.appendleft((y, x))
            elif self.direction == (0, -1):
                x+=1
                self.snake.appendleft((y, x))
            elif self.direction == (1, 0):
                y-=1
                self.snake.appendleft((y, x))
            elif self.direction == (-1, 0):
                y+=1
                self.snake.appendleft((y, x))

    def _generate_food(self):
        y, x = self.np_random.randint(self.height), self.np_random.randint(self.width)
        while self.state[y][x]:
            y, x = self.np_random.randint(self.height), self.np_random.randint(self.width)
        self.state[y][x] = 2

        return y, x

    def _check_for_collision(self, y, x):
        done = False
        pop = True
        reward = self.time_penalty

        if self.state[y][x]:
            if self.state[y][x] == 2:
                pop = False
                reward += (self.food_reward+self.snake_size-3)
                self.snake_size += 1
                if self.snake_size == self.max_size:
                    reward += self.win_reward
                    self.game_over = done = True
                self.food = self._generate_food()
            else:
                reward += self.loss_penalty
                self.game_over = done = True
                pop = False

        self.state[y][x] = 1

        return reward, done, pop

    def step(self, action):
        y, x = self.snake[-1]
        if action == 0: # straight
            y += self.direction[0]
            x += self.direction[1]
        elif action == 1: # right
            if self.direction[0] == 0:
                self.direction = (-self.direction[1], 0)
                y += self.direction[0]
            else:
                self.direction = (0, self.direction[0])
                x += self.direction[1]
        elif action == 2: # left
            if self.direction[0] == 0:
                self.direction = (self.direction[1], 0)
                y += self.direction[0]
            else:
                self.direction = (0, -self.direction[0])
                x += self.direction[1]
        else:
            raise ValueError("Action can only be 0, 1 or 2")

        if self.game_over:
            raise RuntimeError("You're calling step() even though the environment has returned done = True."
                               "You should restart the environment after receiving done = True")
        reward, done, pop = self._check_for_collision(y, x)

        if not done:
            self.snake.append((y, x))

        if pop:
            y, x = self.snake.popleft()
            self.state[y][x] = 0

        observation = self.state

        info = {
            "snake": self.snake,
            "snake_size": self.snake_size,
            "direction": self.direction,
            "food": self.food
        }

        return observation, reward, done, info

    def reset(self):
        self.game_over = False
        dir_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        direction = dir_list[random.randint(0, 3)]
        self.direction = direction

        while self.snake:
            y, x = self.snake.pop()
            self.state[y][x] = 0

        self.state[self.food[0]][self.food[1]] = 0
        
        if direction == (1, 0): # up
            self.starting_position = [random.randint(self.snake_size+1, self.height-1), random.randint(1, self.width-1)]
        elif direction == (-1, 0): # down
            self.starting_position = [random.randint(1, self.height-self.snake_size), random.randint(1, self.width-1)]
        elif direction == (0, 1): # right
            self.starting_position = [random.randint(1, self.height-1), random.randint(self.snake_size+1, self.width-1)]
        elif direction == (0, -1): # left
            self.starting_position = [random.randint(1, self.height-1), random.randint(1, self.width-self.snake_size)]
        print(self.direction, self.starting_position)
        self._init_field(self.starting_position, self.initial_size)
        print(self.snake)
        self.food = self._generate_food()
        self.snake_size = self.initial_size

        return self.state

    def _to_rgb(self, scaling_factor):
        scaled_grid = np.zeros(((self.height + 2) * scaling_factor, (self.width + 2) * scaling_factor), dtype="uint8")
        scaled_grid[:, :scaling_factor] = scaled_grid[:, -scaling_factor:] = 255
        scaled_grid[:scaling_factor, :] = scaled_grid[-scaling_factor:, :] = 255

        y, x = self.food
        scaled_y, scaled_x = y * scaling_factor, x * scaling_factor
        scaled_grid[scaled_y : scaled_y + scaling_factor, scaled_x : scaled_x + scaling_factor] = 255

        for (y, x) in self.snake:
            scaled_y, scaled_x = y * scaling_factor, x * scaling_factor
            scaled_grid[scaled_y : scaled_y + scaling_factor, scaled_x : scaled_x + scaling_factor] = 255

        img = np.empty(((self.height + 2) * scaling_factor, (self.width + 2) * scaling_factor, 3), dtype="uint8")
        img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = scaled_grid

        return img

    def render(self, mode="human", close=False):
        img = self._to_rgb(self.scaling_factor)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            time.sleep(0.027)

            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
