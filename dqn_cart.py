import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import random
import numpy as np
import gym
from collections import deque

GAMMA = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay_rate = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPISODES = 5000
MAX_FRAMES_PLAYED = 500
ACTION_SIZE = 2 # Cartpole specific: can move cart left or right
STATE_SIZE = 4 # Cartpole specific: state = [cart pos, cart vel, pole angle, pole tip vel]

evals = []

def evaluate(model, games=100):
    env = gym.make('CartPole-v1')
    total_score = 0
    for episode in range(games):
        state = env.reset()
        state = np.reshape(state, [1, STATE_SIZE])
        score = 0
        done = False
        while not done:
            action = np.argmax(model.predict(state)[0])

            new_state, reward, done, info = env.step(action)

            score += reward
            state = new_state
            state = np.reshape(state, [1, STATE_SIZE])

        total_score += score
    return total_score / games

model = Sequential()

# The input to the neural network is the state [pos, vel, angle, tip vel], and the output is
# the estimated rewards of doing each action [left, right]
model.add(Dense(24, input_dim=STATE_SIZE, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(ACTION_SIZE, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

memory = deque(maxlen=2000) # only save the last 2000 frames
env = gym.make('CartPole-v1')
state = None

for episode in range(NUM_EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, STATE_SIZE]) # reshape to what Keras expects

    for t in range(MAX_FRAMES_PLAYED):
        #env.render()

        # Act
        action = None
        # if < E, act randomly to explore, otherwise be greedy and choose predicted optimal action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            predicted_reward = model.predict(state)
            action = np.argmax(predicted_reward[0]) # choose action with highest predicted reward

        # Update game with selected action
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, STATE_SIZE])

        # record the the new state and reward achieved
        memory.append((state, action, reward, next_state, done))

        # Update current state to the new one
        state = next_state

        # if gameover, then stop and go to next game
        if done:
            if (episode + 1) % 100 == 0:
                print("episode:", episode)
                #evals.append(evaluate(model))
                #print("game: ", episode + 1, ", eval:", evals[-1])
            #print("episode:", episode, ", score =", t)
            break

    # now do the training on a random batch of recorded states (if enough memory has been saved)
    if BATCH_SIZE > len(memory):
        continue

    # Only train on a random fixed portion of the data. This is fixed because its easier to have
    # a fixed input into a neural network, and it is random to avoid problems of correlated
    # consecutive frames, as well as 
    minibatch = random.sample(memory, BATCH_SIZE)
    # Collect a list of states (inputs) and target rewards (outputs) to have Keras train on
    states, targets_f = [], []
    for state, action, reward, next_state, done in minibatch:
        # According to paper, if you are at terminal state, the target is just the reward,
        # otherwise use the predicted future reward. This is because we want the agent to learn to
        # perform well in the long run by learning to maximize the future reward based on the current state
        target = reward
        if not done:
            # target = (reward achieved) + gamma*(predicted future reward)
            target = reward + GAMMA * np.max(model.predict(next_state)[0])

        # Keras will attempt to learn the reward of each possible action, so set the action taken to
        # have the predicted future reward to learn to maximize it. Approximately mapping
        # the current state to the future reward
        target_f = model.predict(state)
        target_f[0][action] = target

        states.append(state[0])
        targets_f.append(target_f[0])
    
    hist = model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
    loss = hist.history['loss'][0]
    epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)
    
#model.save_weights('cartpole-dqn-1000eps.h5')
fname = "dqn_cartv1-" + str(GAMMA) + str(epsilon_min) + str(epsilon_decay_rate) + str(LEARNING_RATE) + str(BATCH_SIZE) + ".txt"
final = evaluate(model, 1000)
print(fname, ":", final)
f = open(fname, "w")
f.write(str(evals) + "\n" + str(final))
f.close()

