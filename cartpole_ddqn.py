import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import *
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

class PERReplayBuffer:
    # hyper parameters from paper
    E = 0.01
    A = 0.6
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def size(self):
        return self.tree.n_entries

    def isFull(self):
        return self.size() == self.capacity

    def add(self, data, error):
        p = math.pow(error + self.E, self.A)
        self.tree.add(p, data)

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

class UniformReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def size(self):
        return len(self.memory)

    # store as (state, action, reward, newState, done)
    def add(self, data):
        self.memory.append(data)

    def isFull(self):
        return self.size() == self.capacity

    def sample(self, size):
        return random.sample(self.memory, size)

class Network:
    def __init__(self, learningRate, inputSize, outputSize):
        self.learningRate = learningRate
        self.inputSize = inputSize
        self.outputSize= outputSize
        self.model = self.buildModel()

    def buildModel(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.inputSize, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.outputSize, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learningRate))
        return model

    def predict(self, xs):
        return self.model.predict(xs)

    def predictOne(self, x):
        return self.model.predict(x.reshape(1, self.inputSize)).flatten()
    
    def train(self, xs, ys):
        self.model.fit(xs, ys, epochs=1, verbose=0)

    def setWeights(self, otherNetwork):
        self.model.set_weights(otherNetwork.model.get_weights())

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

class Agent:
    def __init__(self, currentNet, targetNet, buffer, gamma = 0.99, minEpsilon = 0.1, decayRate = .999, batchSize = 32, tau = 10000):
        self.replayBuffer = buffer
        self.GAMMA = gamma
        self.MIN_EPSILON = minEpsilon
        self.EPSILON_DECAY_RATE = decayRate
        self.BATCH_SIZE = batchSize
        self.TAU = tau

        self.currentNetwork = currentNet
        self.targetNetwork  = targetNet
        self.steps = 0
        self.epsilon = 1

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.currentNetwork.outputSize - 1)
        else:
            return np.argmax(self.currentNetwork.predictOne(state))

    def updateTargetModel(self):
        self.targetNetwork.setWeights(self.currentNetwork)

    def observe(self, state, action, reward, newState, done):
        # calculate new sample error
        currentPredictions = self.currentNetwork.predictOne(state)
        newPredictions = self.targetNetwork.predictOne(newState)
        target = reward
        if not done:
            target += self.GAMMA * newPredictions[np.argmax(newPredictions)]

        error = abs(currentPredictions[action] - target)

        self.replayBuffer.add((state, action, reward, newState, done), error)
        self.steps += 1
        self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY_RATE)

        if self.steps % self.TAU == 0:
            self.updateTargetModel()

    def experienceReplay(self):
        # sample random batch from replay memory
        if self.BATCH_SIZE > self.replayBuffer.size():
            return
        batch = self.replayBuffer.sample(self.BATCH_SIZE)

        # combine all of the BATCH_SIZE states and new states into two lists
        # to let keras do the predictions in two calls
        states = np.array([x[1][0] for x in batch])
        newStates = np.array([x[1][3] for x in batch])
        currentPredictions = self.currentNetwork.predict(states)
        newPredictions = self.targetNetwork.predict(newStates)
        
        # Create list of xs and ys do train all of the batch items at once
        xs = np.zeros((self.BATCH_SIZE, self.currentNetwork.inputSize))
        ys = np.zeros((self.BATCH_SIZE, self.currentNetwork.outputSize))

        for i in range(self.BATCH_SIZE):
            index, (state, action, reward, newState, done) = batch[i]
            target = reward
            if not done:
                target += self.GAMMA * newPredictions[i][np.argmax(newPredictions[i])]

            # Approximately map current Q to the target Q
            currentQs = currentPredictions[i]
            oldQ = currentQs[action]
            currentQs[action] = target
            xs[i] = states[i] 
            ys[i] = currentQs

            # get errors for PER
            error = abs(oldQ - target)
            self.replayBuffer.update(index, error)

        # update network / fit the model
        self.currentNetwork.train(xs, ys)

class RandomAgent:
    def __init__(self, buffer):
        self.replayBuffer = buffer
        self.steps = 0

    def selectAction(self, state):
        return random.randint(0, 1)

    def observe(self, state, action, reward, newState, done):
        error = abs(reward)
        self.replayBuffer.add((state, action, reward, newState, done), error)
        self.steps += 1

    def experienceReplay(self):
        pass


class Environment:
    def __init__(self, name):
        self.name = name
        self.stateSpaceSize  = 0
        self.actionSpaceSize = 0
        self.env = gym.make(name)
        self.scores = []

    def save(self, scoresPath, figPath):
        s = str(self.scores)
        s = s[1:-1]
        f = open(scoresPath, "w")
        f.write(s)
        f.close()
        plt.plot(list(range(len(self.scores)))[::2], self.scores[::2])
        plt.savefig(figPath)

    def plot(self):
        plt.plot(list(range(len(self.scores)))[::2], self.scores[::2])
        plt.show()

    def reset(self):
        s = self.env.reset()
        return self.processState(s)

    def processState(self, state):
        return state

    def stepEnvironment(self, action):
        newState, reward, done, info = self.env.step(action)
        newState = self.processState(newState)
        return newState, reward, done, info

    def run(self, agent, numEpisodes, verbose = True, render = False):
        for episode in range(numEpisodes):
            state = self.processState(self.env.reset())
            score = 0            
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # select action
                action = agent.selectAction(state)

                # execute that action
                newState, reward, done, info = self.stepEnvironment(action)
                reward = reward if not done or score == 499 else -100
                
                # store experience in replay memory
                agent.observe(state, action, reward, newState, done)
                agent.experienceReplay()

                # observe reward and next state
                score += reward
                state = newState

            score = score if score == 500 else score + 100
            self.scores.append(score)
            if verbose:
                print("Episode", episode, ":", score)

class CartPole(Environment):
    def __init__(self):
        super().__init__('CartPole-v1')
        self.actionSpaceSize = 2
        self.stateSpaceSize = 4


env = CartPole()
replayBuffer = PERReplayBuffer(200000)
randomAgent = RandomAgent(replayBuffer)
while not replayBuffer.isFull():
    env.run(randomAgent, 100, False, False)
    print(replayBuffer.size() / 200000)
print("full")

tNetwork = Network(0.00025, env.stateSpaceSize, env.actionSpaceSize)
cNetwork = Network(0.00025, env.stateSpaceSize, env.actionSpaceSize)
agent = Agent(cNetwork, tNetwork, replayBuffer)

env.run(agent, 10000)
env.save("results/dqqn_scores.txt", "results/ddqn_scores.png")
