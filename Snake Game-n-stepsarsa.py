"""
Snake Eater
Made with PyGame
https://github.com/rajatdiptabiswas/snake-pygame
n-step sarsa
rewards -1 for death, +100 for food
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
frame_size_x = 300
frame_size_y = 150

#for training algorithm, dont show GFX
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


# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True

direction = 2
change_to = direction

score = 0

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

#uses epsilon greedy to determine which action to take in a given state, e gets larger as episodes increase, ties are broken randomly
def policy(SA, QA, count):
    r = random.randrange(1, 101)
    e = count/10000
    if r > e:
       action = random.randrange(0, 4)
    else:
        action = np.random.choice(np.where(QA[SA[0], SA[1], SA[2], SA[3], SA[4], SA[5], SA[6], :] == QA[SA[0], SA[1], SA[2], SA[3], SA[4], SA[5], SA[6], :].max())[0])
    return action

X = []
Y = []

eCount = 0
gamma = 0.5
alpha = 0.5
Q = np.zeros([3, 3, 2, 2, 2, 2, 4, 4])
model = np.zeros((3, 3, 2, 2, 2, 2, 4, 8), dtype=object)
n = 5
xCount = 0
for k in range(0, 10000):
    eCount+=1
    print(eCount)
    #intialize action, random snake head/body postion
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
    #spawn food in random spot
    food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
    food_spawn = True
    direction = A
    pairs = []
    states = []
    actions = []
    rewards = []

    #pick first action and append
    positions = [[snake_pos[0], snake_pos[1]+10], [snake_pos[0]-10, snake_pos[1]], [snake_pos[0]+10, snake_pos[1]], [snake_pos[0], snake_pos[1]-10]]
    val = np.array([1, 1, 1, 1])
    for x in range(0, 4):
        if positions[x][0] < 0 or positions[x][0] > frame_size_x-10:
            val[x] = 0
        if positions[x][1] < 0 or positions[x][1] > frame_size_y-10:
            val[x] = 0
        # Touching the snake body
        for block in snake_body[1:]:
            if positions[x][0] == block[0] and positions[x][1] == block[1]:
                val[x] = 0
    #first state and append
    S = [np.sign(food_pos[0]-snake_pos[0])+1, np.sign(snake_pos[1]-food_pos[1])+1, val[0], val[1], val[2], val[3], A]
    states.append(S)
    actions.append(A)
    T = np.infty
    t = -1

    #intialize event
    for event in pygame.event.get():
        newEvent = pygame.event.Event(pygame.KEYDOWN, unicode="b", key=pygame.K_a, mod=pygame.KMOD_NONE)
        pygame.event.post(newEvent)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Whenever a key is pressed down
        elif event.type == pygame.KEYDOWN:
            #get action
            change_to = policy(S, Q, eCount)
    #set action
    A = change_to
    actions.append(A)
    # Main logic, each step
    while True:
        t+=1
        reward = 0
        change_to = A
        if t<T:
            #dont let snake instantly go other direction
            if change_to == 0 and direction != 3:
                direction = 0
            if change_to == 3 and direction != 0:
                direction = 3
            if change_to == 1 and direction != 2:
                direction = 1
            if change_to == 2 and direction != 1:
                direction = 2

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
                reward = 10
                score += 1
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
            # Touching the snake body
            for block in snake_body[1:]:
                if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                    reward = -1
            if reward == -1:
                X.append(score)
                if eCount%100 == 0:
                    Y.append(np.mean(X))
                    X.clear()
            #set reward
            R = reward
            #get s prime
            positions = [[snake_pos[0], snake_pos[1]+10], [snake_pos[0]-10, snake_pos[1]], [snake_pos[0]+10, snake_pos[1]], [snake_pos[0], snake_pos[1]-10]]
            val = np.array([1, 1, 1, 1])
            for x in range(0, 4):
                if positions[x][0] < 0 or positions[x][0] > frame_size_x-10:
                    val[x] = 0
                if positions[x][1] < 0 or positions[x][1] > frame_size_y-10:
                    val[x] = 0
                # Touching the snake body
                for block in snake_body[1:]:
                    if positions[x][0] == block[0] and positions[x][1] == block[1]:
                        val[x] = 0
            S = [np.sign(food_pos[0]-snake_pos[0])+1, np.sign(snake_pos[1]-food_pos[1])+1, val[0], val[1], val[2], val[3], A]

            #append sp and reward
            states.append(S)
            rewards.append(R)
            if R == -1:
                T = t+1
            else:
                #intialize event
                for event in pygame.event.get():
                    newEvent = pygame.event.Event(pygame.KEYDOWN, unicode="b", key=pygame.K_a, mod=pygame.KMOD_NONE)
                    pygame.event.post(newEvent)
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    # Whenever a key is pressed down
                    elif event.type == pygame.KEYDOWN:
                        #get action
                        change_to = policy(S, Q, eCount)
                #set action
                A = change_to
                actions.append(A)

            #shows score
            show_score(20, white, 'consolas', 20)
            # Refresh game screen
            pygame.display.update()
            #if eCount > 3000:
                #difficulty = 20
            # Refresh rate
            fps_controller.tick(difficulty)
        tao = t-n+1
        if tao >= 0:
            G = 0
            for i in range(tao+1, min(tao+n, T)):
                G+=gamma**(i-tao-1)*rewards[i]
            if tao+n < T:
                G+= gamma**n * Q[states[tao+n][0], states[tao+n][1], states[tao+n][2], states[tao+n][3], states[tao+n][4], states[tao+n][5], states[tao+n][6], actions[tao+n]]
            Q[states[tao][0], states[tao][1], states[tao][2], states[tao][3], states[tao][4], states[tao][5], states[tao][6], actions[tao]] += alpha*(G-Q[states[tao][0], states[tao][1], states[tao][2], states[tao][3], states[tao][4], states[tao][5], states[tao][6], actions[tao]])
        if tao == T-1:
            break
fig = plt.figure()
plt.plot(range(1, np.asarray(Y).shape[0]+1), Y)
plt.ylabel('average score')
plt.xlabel('hundredth iteration')
fig.suptitle('n-step SARSA', fontsize=20)
fig.savefig('sarsa.jpg')
plt.show()