import gym
import numpy as np
import random
import pylab as plt

from gym import spaces

'''
action space: forward 0, left 1, right 2 (relative to current direction)
direction: right (1, 0), left (-1, 0), up (0, 1), down (0, -1)
observation space: 
    x_dim+2 by y_dim+2 box
    0 is open space
    1 is border block
    1 is body
    2 is head
    3 is food
'''

class SnakeEnv(gym.Env):
    def __init__(self, x_dim=15, y_dim=15, snake_size=3):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=[10])
        self.max_size = x_dim * y_dim
        self.food_reward = 10
        self.time_reward = -1/(self.max_size)
        self.win_reward = 1000
        self.loss_penalty = -10
        self.initial_snake_size = snake_size

        self.game_over = False
        self.snake_size = self.initial_snake_size
        # randomize initial position and direction
        dir_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.direction = dir_list[random.randint(0, 3)]
        if self.direction == (0, 1): # up
            self.starting_position = [random.randint(1, x_dim), random.randint(snake_size+1, y_dim)]
        elif self.direction == (0, -1): # down
            self.starting_position = [random.randint(1, x_dim), random.randint(1, y_dim-snake_size)]
        elif self.direction == (1, 0): # right
            self.starting_position = [random.randint(snake_size+1, x_dim), random.randint(1, y_dim)]
        elif self.direction == (-1, 0): # left
            self.starting_position = [random.randint(1, x_dim-snake_size), random.randint(1, y_dim)]

        # initialize state and add snake to it
        self.state = np.zeros((x_dim+2, y_dim+2))
        self.snake = []
        self.snake.append(self.starting_position)
        self.state[self.starting_position[0], self.starting_position[1]] = 2
        x, y = self.starting_position
        for i in range(1, snake_size):
            if self.direction == (1, 0):
                x-=1
                self.snake.append([x, y])
            elif self.direction == (-1, 0):
                x+=1
                self.snake.append([x, y])
            elif self.direction == (0, 1):
                y-=1
                self.snake.append([x, y])
            elif self.direction == (0, -1):
                y+=1
                self.snake.append([x, y])
            self.state[x][y] = 1

        # add boundaries to state
        self.state[:, 0] = self.state[:, -1] = 1
        self.state[0, :] = self.state[-1, :] = 1

        # add food to state and self variable
        self.food_position = [random.randint(1, x_dim), random.randint(1, y_dim)]
        while self.food_position in self.snake:
            self.food_position = [random.randint(1, x_dim), random.randint(1, y_dim)]
        self.state[self.food_position[0], self.food_position[1]] = 3

        self.im = plt.imshow(self.state, cmap = 'cubehelix', interpolation='none',vmin=0,vmax=2)  

    def check_for_collision(self, x, y):
        done = False
        reward = self.time_reward # time step decrement

        if self.state[x][y] == 3: # ate food
            reward += (self.snake_size-2)*self.food_reward
            self.snake_size += 1
            self.state[self.snake[0][0], self.snake[0][1]] = 1
            self.snake.insert(0, [x, y])
            if self.state[x][y]==1: # collision
                reward += self.loss_penalty
                self.game_over = done = True
            self.state[self.snake[0][0], self.snake[0][1]] = 2
            if self.snake_size == self.max_size:
                reward += self.win_reward
                self.game_over = done = True
            self.food_position = [random.randint(1, self.x_dim), random.randint(1, self.y_dim)]
            while self.food_position in self.snake:
                self.food_position = [random.randint(1, self.x_dim), random.randint(1, self.y_dim)]
            self.state[self.food_position[0], self.food_position[1]] = 3
        else: # normal progression
            self.state[self.snake[0][0], self.snake[0][1]] = 1
            self.snake.insert(0, [x, y])
            self.state[self.snake[-1][0], self.snake[-1][1]] = 0
            self.snake.pop()
            if self.state[x][y]==1: # collision
                reward += self.loss_penalty
                self.game_over = done = True
            self.state[self.snake[0][0], self.snake[0][1]] = 2

        return reward, done

    def step(self, action):
        x, y = self.snake[0]
        if action == 1:
            x += self.direction[0]
            y += self.direction[1]
        elif action == 0:
            if self.direction[0] == 0:
                x -= self.direction[1]
                self.direction = (-self.direction[1], 0)
            else:
                y += self.direction[0]
                self.direction = (0, self.direction[0])
        elif action == 2:
            if self.direction[0] == 0:
                x += self.direction[1]
                self.direction = (self.direction[1], 0)
            else:
                y -= self.direction[0]
                self.direction = (0, -self.direction[0])
        else:
            raise ValueError("Action can only be 0, 1 or 2")

        reward, done = self.check_for_collision(x, y)
        to_return = [0]*3
        relative_food_dir = [0]*4 # right, left, infront, behind
        if not done:
            if self.direction==(1,0):
                to_return[0] = self.state[x,y+1]
                to_return[1] = self.state[x+1,y]
                to_return[2] = self.state[x,y-1]
                if self.food_position[0]-x>0:
                    relative_food_dir[2] = 1
                elif self.food_position[0]-x<0:
                    relative_food_dir[3] = 1
                if self.food_position[1]-y>0:
                    relative_food_dir[1] = 1
                elif self.food_position[1]-y<0:
                    relative_food_dir[0] = 1
                    
            elif self.direction==(-1,0):
                to_return[0] = self.state[x,y-1]
                to_return[1] = self.state[x-1,y]
                to_return[2] = self.state[x,y+1]
                if self.food_position[0]-x<0:
                    relative_food_dir[2] = 1
                elif self.food_position[0]-x>0:
                    relative_food_dir[3] = 1
                if self.food_position[1]-y<0:
                    relative_food_dir[1] = 1
                elif self.food_position[1]-y>0:
                    relative_food_dir[0] = 1

            elif self.direction==(0,1):
                to_return[0] = self.state[x-1,y]
                to_return[1] = self.state[x,y+1]
                to_return[2] = self.state[x+1,y]
                if self.food_position[1]-y>0:
                    relative_food_dir[2] = 1
                elif self.food_position[1]-y<0:
                    relative_food_dir[3] = 1
                if self.food_position[0]-x<0:
                    relative_food_dir[1] = 1
                elif self.food_position[0]-x>0:
                    relative_food_dir[0] = 1

            elif self.direction==(0,-1):
                to_return[0] = self.state[x+1,y]
                to_return[1] = self.state[x,y-1]
                to_return[2] = self.state[x-1,y]
                if self.food_position[1]-y<0:
                    relative_food_dir[2] = 1
                elif self.food_position[1]-y>0:
                    relative_food_dir[3] = 1
                if self.food_position[0]-x>0:
                    relative_food_dir[1] = 1
                elif self.food_position[0]-x<0:
                    relative_food_dir[0] = 1

            else:
                print("unkown direction")

        info = {
            "snake_size": self.snake_size,
            "reward": reward
        }
        # print(f"action: {action}")
        # print(f"direction: {self.direction}")
        # print(f"reward: {reward}")
        # print(f"snake: {self.snake}")
        for i, v in enumerate(to_return):
            if v==3:
                to_return[i] = 2

        return to_return, relative_food_dir, reward, done, info
        
    def reset(self):
        self.game_over = False
        self.snake_size = self.initial_snake_size
        # randomize initial position and direction
        dir_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.direction = dir_list[random.randint(0, 3)]
        if self.direction == (0, 1): # up
            self.starting_position = [random.randint(1, self.x_dim), random.randint(self.snake_size+1, self.y_dim)]
        elif self.direction == (0, -1): # down
            self.starting_position = [random.randint(1, self.x_dim), random.randint(1, self.y_dim-self.snake_size)]
        elif self.direction == (1, 0): # right
            self.starting_position = [random.randint(self.snake_size+1, self.x_dim), random.randint(1, self.y_dim)]
        elif self.direction == (-1, 0): # left
            self.starting_position = [random.randint(1, self.x_dim-self.snake_size), random.randint(1, self.y_dim)]

        # initialize state and add snake to it
        self.state = np.zeros((self.x_dim+2, self.y_dim+2))
        self.snake = []
        self.snake.append(self.starting_position)
        self.state[self.starting_position[0], self.starting_position[1]] = 2
        x, y = self.starting_position
        for i in range(1, self.snake_size):
            if self.direction == (1, 0):
                x-=1
                self.snake.append([x, y])
            elif self.direction == (-1, 0):
                x+=1
                self.snake.append([x, y])
            elif self.direction == (0, 1):
                y-=1
                self.snake.append([x, y])
            elif self.direction == (0, -1):
                y+=1
                self.snake.append([x, y])
            self.state[x][y] = 1

        # add boundaries to state
        self.state[:, 0] = self.state[:, -1] = 1
        self.state[0, :] = self.state[-1, :] = 1

        # add food to state and self variable
        self.food_position = [random.randint(1, self.x_dim), random.randint(1, self.y_dim)]
        while self.food_position in self.snake:
            self.food_position = [random.randint(1, self.x_dim), random.randint(1, self.y_dim)]
        self.state[self.food_position[0], self.food_position[1]] = 3

        to_return = [0]*3
        relative_food_dir = [0]*4 # right, left, infront, behind
        if self.direction==(1,0):
            to_return[0] = self.state[x,y+1]
            to_return[1] = self.state[x+1,y]
            to_return[2] = self.state[x,y-1]
            if self.food_position[0]-x>0:
                relative_food_dir[2] = 1
            elif self.food_position[0]-x<0:
                relative_food_dir[3] = 1
            if self.food_position[1]-y>0:
                relative_food_dir[1] = 1
            elif self.food_position[1]-y<0:
                relative_food_dir[0] = 1
                
        elif self.direction==(-1,0):
            to_return[0] = self.state[x,y-1]
            to_return[1] = self.state[x-1,y]
            to_return[2] = self.state[x,y+1]
            if self.food_position[0]-x<0:
                relative_food_dir[2] = 1
            elif self.food_position[0]-x>0:
                relative_food_dir[3] = 1
            if self.food_position[1]-y<0:
                relative_food_dir[1] = 1
            elif self.food_position[1]-y>0:
                relative_food_dir[0] = 1

        elif self.direction==(0,1):
            to_return[0] = self.state[x-1,y]
            to_return[1] = self.state[x,y+1]
            to_return[2] = self.state[x+1,y]
            if self.food_position[1]-y>0:
                relative_food_dir[2] = 1
            elif self.food_position[1]-y<0:
                relative_food_dir[3] = 1
            if self.food_position[0]-x<0:
                relative_food_dir[1] = 1
            elif self.food_position[0]-x>0:
                relative_food_dir[0] = 1

        elif self.direction==(0,-1):
            to_return[0] = self.state[x+1,y]
            to_return[1] = self.state[x,y-1]
            to_return[2] = self.state[x-1,y]
            if self.food_position[1]-y<0:
                relative_food_dir[2] = 1
            elif self.food_position[1]-y>0:
                relative_food_dir[3] = 1
            if self.food_position[0]-x>0:
                relative_food_dir[1] = 1
            elif self.food_position[0]-x<0:
                relative_food_dir[0] = 1

        else:
            print("unkown direction")

        for i, v in enumerate(to_return):
            if v==3:
                to_return[i] = 2

        return to_return, relative_food_dir

    def render(self, mode='human', close=False):
        # render the environment to the screen  
        img = np.rot90(self.state[1:self.x_dim+1, 1:self.y_dim+1])
        #img = np.where(img != 0, 1, img)
        self.im.set_data(img)
        plt.axis('off')
        plt.draw()
        plt.pause(0.0001)