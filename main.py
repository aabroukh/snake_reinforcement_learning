"""
Snake Eater
Made with PyGame
initial snake game code taken from: 
https://github.com/rajatdiptabiswas/snake-pygame
"""

import pygame, sys, time, random
import numpy as np
import heapq as hq
import matplotlib.pyplot as plt


# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
difficulty = 120

# Window size
frame_size_x = 1080
frame_size_y = 720

# for training algorithm, dont show GFX
training = False

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')

# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))

# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()

# Score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x/10, 15)
    else:
        score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
    game_window.blit(score_surface, score_rect)

# uses epsilon greedy to determine which action to take in a given state, e gets larger as episodes increase, ties are broken randomly
def policy():
    return random.randrange(0, 4)

# graph variables
X = []
Y = []

# Game variables
food_spawn = True
change_to = None
score = 0

# number of episodes 
for eCount in range(1, 10001):
    print(eCount)
    # initialize action, random snake head/body postion
    A = random.randrange(0, 4)
    if A == 0:
        rand_x = random.randint(0, frame_size_x//10)*10
        rand_y = random.randint(3, frame_size_y//10)*10
        snake_pos = [rand_x, rand_y]
        snake_body = [[snake_pos[0], snake_pos[1]], [snake_pos[0], snake_pos[1]-10], [100, snake_pos[1]-(2*10)]]
    elif A == 1:
        rand_x = random.randint(0, frame_size_x//10-3)*10
        rand_y = random.randint(0, frame_size_y//10)*10
        snake_pos = [rand_x, rand_y]
        snake_body = [[snake_pos[0], snake_pos[1]], [snake_pos[0]+10, snake_pos[1]], [100+(2*10), snake_pos[1]]]
    elif A == 2:
        rand_x = random.randint(3, frame_size_x//10)*10
        rand_y = random.randint(0, frame_size_y//10)*10
        snake_pos = [rand_x, rand_y]
        snake_body = [[snake_pos[0], snake_pos[1]], [snake_pos[0]-10, snake_pos[1]], [100-(2*10), snake_pos[1]]]
    else:
        rand_x = random.randint(0, frame_size_x//10)*10
        rand_y = random.randint(0, frame_size_y//10-3)*10
        snake_pos = [rand_x, rand_y]
        snake_body = [[snake_pos[0], snake_pos[1]], [snake_pos[0], snake_pos[1]+10], [100, snake_pos[1]+(2*10)]]
    score = 0

    # spawn food in random spot
    food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
    food_spawn = True
    direction = A
    pairs = []

    # Main logic, each step
    while True:
        # set reward to 0 unless gets food or hits wall
        reward = 0
        # initialize event
        for event in pygame.event.get():
            newEvent = pygame.event.Event(pygame.KEYDOWN, unicode="b", key=pygame.K_a, mod=pygame.KMOD_NONE)
            pygame.event.post(newEvent)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Whenever a key is pressed down
            elif event.type == pygame.KEYDOWN:
                # get action
                change_to = policy()
        # set action
        A = change_to

        # dont let snake instantly go other direction
        if change_to == 0 and direction != 3:
            direction = 0
        if change_to == 1 and direction != 2:
            direction = 1
        if change_to == 2 and direction != 1:
            direction = 2
        if change_to == 3 and direction != 0:
            direction = 3

        # Moving the snake
        if direction == 0:
            snake_pos[1] -= 10
        if direction == 3:
            snake_pos[1] += 10
        if direction == 1:
            snake_pos[0] -= 10
        if direction == 2:
            snake_pos[0] += 10

        # Snake body growing mechanism
        snake_body.insert(0, list(snake_pos))
        if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
            score += 1
            reward = score*10
            food_spawn = False
        else:
            snake_body.pop()

        # Spawning food on the screen
        if not food_spawn:
            food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
        food_spawn = True

        # GFX
        game_window.fill(black)
        for pos in snake_body:
            # Snake body
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if snake_pos[0] < 0 or snake_pos[0] > frame_size_x-10:
            reward = -1
        if snake_pos[1] < 0 or snake_pos[1] > frame_size_y-10:
            reward = -1

        # initialize input array, 1=hit object, 2=head, 3=food
        input = np.zeros([frame_size_y+2, frame_size_x+2, 3])
        input[0, :] = [1, 0, 0]
        input[frame_size_y+1, :] = [1, 0, 0]
        input[:, 0] = [1, 0, 0]
        input[:, frame_size_x+1] = [1, 0, 0]
        input[snake_pos[1]+1, snake_pos[0]+1] = [0, 1, 0]
        input[food_pos[1]+1, food_pos[0]+1] = [0, 0, 1]

        # Touching the snake body & add body to input array
        for block in snake_body[1:]:
            input[block[1]+1, block[0]+1] = [1, 0, 0]
            if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                reward = -1

        # model part
        
        
        if reward == -1:
            break
        
        #shows score
        show_score(20, white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        fps_controller.tick(difficulty)
    # keep track of progress
    X.append(score)
    if eCount%100 == 0:
        Y.append(np.mean(X))
        X.clear()
# plot and make image of progress
fig = plt.figure()
plt.plot(range(1, np.asarray(Y).shape[0]+1), Y)
plt.ylabel('average score')
plt.xlabel('hundredth iteration')
fig.suptitle('tabular-DQ', fontsize=20)
fig.savefig('DQ.jpg')
plt.show()