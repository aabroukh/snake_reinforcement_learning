import gym
import gym_snake
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras import layers

env = gym.make('gym-snake:snake-v0')

# Configuration paramaters for the whole setup
gamma = 0.97  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 1000

num_actions = 3
def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(env.x_dim+2, env.y_dim+2, 1))

    # Convolutions on the frames on the screen
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    layer1 = layers.Conv2D(256, 3, strides=3, activation="relu", kernel_initializer=initializer)(inputs)
    layer2 = layers.Flatten()(layer1)
    layer3 = layers.Dense(128, activation="relu")(layer2)
    action = layers.Dense(num_actions, activation="softmax")(layer3)

    return keras.Model(inputs=inputs, outputs=action)
loading = False
if not loading:
    model = create_q_model()
    model_target = create_q_model()
else:
    model = create_q_model()
    model_target = create_q_model()

optimizer = Adam(learning_rate=0.0025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
score_history = []
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 100000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 100
# Using huber loss for stability
loss_function = keras.losses.MeanSquaredError()
try:     
    while True:
        state = env.reset()
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            env.render()
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
            
            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every frame and once batch size is over 32
            if len(done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Log details
                print(f"episode count: {episode_count}, snake size: {env.snake_size}, episode reward: {episode_reward}")

            if frame_count % update_after_actions == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())

            if done:
                break

        score_history.append(env.snake_size)
        episode_count += 1
        if env.snake_size == env.x_dim*env.y_dim:  # Condition to consider the task solved
            print(f"Solved at episode {episode_count}!")
            break       
except:
    model.save('./saved_models/model_ecount_'+str(episode_count)+'.h5')
    model_target.save('./saved_models/model_target_ecount_'+str(episode_count)+'.h5')

fig, ax = plt.subplots()
ax.plot(range(0, episode_count), score_history)

ax.set(xlabel='episode', ylabel='end score')
ax.grid()
fig.savefig(f"{episode_count}.png")
plt.show()