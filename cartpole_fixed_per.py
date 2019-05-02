import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import random
import numpy as np
import math
import gym
from collections import deque
import matplotlib.pyplot as plt


# sum tree code from: https://github.com/rlcode/per/blob/master/SumTree.py
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class Memory:
    # hyper parameters from paper
    E = 0.01
    A = 0.6
    def __init__(self, size):
        self.tree = SumTree(size)

    def remember(self, state, action, reward, new_state, done, error):
        p = math.pow(error + self.E, self.A)
        self.tree.add(p, (state, action, reward, new_state, done))

    def sample(self, N):
        batch = []
        binSize = self.tree.total() / N
        for i in range(N):
            s = random.uniform(binSize * i, binSize * (i + 1))
            (index, p, sarsd) = self.tree.get(s)
            batch.append((index, sarsd))

        return batch

    def update(self, index, error):
        p = math.pow(error + self.E, self.A)
        self.tree.update(index, p)

    def size(self):
        return self.tree.n_entries
    

class DQNParameters:
    def __init__(self):
        self.learning_rate = 0.001
        self.gamma = .95
        self.min_epsilon = .01
        self.epsilon_decay_rate = .995
        self.num_episodes = 500
        self.max_time_steps = 10**6
        self.replayMemorySize = 1e6
        self.batch_size = 32
        self.target_update_steps = 100

class CartPoleDQN:
    def __init__(self, params=DQNParameters()):
        self.env = gym.make('CartPole-v1')
        self.STATE_SIZE = self.env.observation_space.shape[0]
        self.ACTION_SIZE  = self.env.action_space.n
        self.GAMMA = params.gamma
        self.MIN_EPSILON = params.min_epsilon
        self.EPSILON_DECAY_RATE = params.epsilon_decay_rate
        self.NUM_EPISODES = params.num_episodes
        self.MAX_TIME_STEPS = params.max_time_steps
        self.replayMemory = Memory(params.replayMemorySize)
        self.BATCH_SIZE = params.batch_size
        self.LEARNING_RATE = params.learning_rate
        self.TARGET_UPDATE_STEPS = params.target_update_steps
        self.scores = []

        self.epsilon = 1.0
        self.model = self.buildModel(params.learning_rate)
        self.targetModel = self.buildModel(params.learning_rate)

    def buildModel(self, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_dim=self.STATE_SIZE, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def updateTargetModel(self):
        self.targetModel.set_weights(self.model.get_weights())

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            predictedQ = self.model.predict(state)
            return np.argmax(predictedQ[0])

    def getSampleError(self, state, action, reward, new_state, done):
        target = reward
        if not done:
            target += self.GAMMA * np.amax(self.targetModel.predict(new_state)[0])

        # Approximately map current Q to new Q
        currentQs = self.model.predict(state)
        return abs(currentQs[0][action] - target)

    def experienceReplay(self):
        # sample random batch from replay memory
        #if len(self.replayMemory) < self.BATCH_SIZE:
        #    return
        #batch = random.sample(self.replayMemory, self.BATCH_SIZE)

        if self.replayMemory.size() < self.BATCH_SIZE:
            return
        batch = self.replayMemory.sample(self.BATCH_SIZE)
        
        # Preprocess states from batch
        states, targets = [], []
        for index, (state, action, reward, new_state, done) in batch:
            target = reward
            if not done:
                target += self.GAMMA * np.amax(self.model.predict(new_state)[0])

            # Approximately map current Q to new Q
            currentQs = self.model.predict(state)
            prediction = currentQs[0][action]
            currentQs[0][action] = target

            states.append(state[0])
            targets.append(currentQs[0])

            # update memory priority
            new_error = abs(prediction - target)
            self.replayMemory.update(index, new_error)

        # update network and epsilon
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY_RATE)
        
    def train(self, numEpisodes=None, render=False):
        if numEpisodes == None:
            numEpisodes = self.NUM_EPISODES

        averageScore = 0
        currentTargetSteps = 0
        
        for episode in range(numEpisodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.STATE_SIZE])
            score = 0
            
            #for timeStep in range(self.MAX_TIME_STEPS):
            while True:
                if render:
                    self.env.render()
                
                # select action
                action = self.selectAction(state)

                # execute that action
                new_state, reward, done, info = self.env.step(action)
                new_state = np.reshape(new_state, [1, self.STATE_SIZE])
                if done:
                    reward = -reward

                # store experience in replay memory
                #self.replayMemory.append((state, action, reward, new_state, done))
                error = self.getSampleError(state, action, reward, new_state, done)
                self.replayMemory.remember(state, action, reward, new_state, done, error)

                # observe reward and next state
                score += reward
                state = new_state
                if done:
                    averageScore += score
                    self.scores.append(score)
                    print("Episode:", episode, " exploration =", self.epsilon, ", score:", score)
                    break
                
                self.experienceReplay()
                #currentTargetSteps += 1
                #if currentTargetSteps > self.TARGET_UPDATE_STEPS:
                #    self.updateTargetModel()
                #    currentTargetSteps = 0

        return averageScore / numEpisodes
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def getFname(self):
        return "dqn_cartv2-" + str(self.GAMMA) + str(self.MIN_EPSILON) + str(self.EPSILON_DECAY_RATE) + \
               str(self.LEARNING_RATE) + str(self.BATCH_SIZE) + ".txt"        

    # Returns average score acheived with current network over 'games' episodes
    def evaluate(self, games=100):
        env = gym.make('CartPole-v1')
        total_score = 0
        for episode in range(games):
            state = env.reset()
            state = np.reshape(state, [1, self.STATE_SIZE])
            score = 0
            done = False
            while not done:
                action = np.argmax(self.model.predict(state)[0])

                new_state, reward, done, info = env.step(action)

                score += reward
                state = new_state
                state = np.reshape(state, [1, self.STATE_SIZE])

            total_score += score
        return total_score / games

params = DQNParameters()
params.learning_rate = .001
params.gamma = .95
params.max_time_steps = 10**6
params.min_epsilon = .01
# params.epsilon_decay_rate = .999997697417558
params.epsilon_decay_rate = .995
params.batch_size = 20
params.replayMemorySize = 1000000

dqn = CartPoleDQN(params)
dqn.train(500)
#for i in range(1, 51):
#    score = dqn.train(100)
#    print("Average score after", 100 * i, "games =", score)

final = dqn.evaluate(100)
print(final)
#plt.plot(games, evals)
#plt.show()
