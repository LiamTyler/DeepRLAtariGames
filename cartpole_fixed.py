import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import random
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt

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
        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
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
    def __init__(self, currNet, targetNet, buffer, gamma = 0.99, minEpsilon = 0.01, decayRate = .999, batchSize = 64, tau = 1000):
        self.replayBuffer = buffer
        self.GAMMA = gamma
        self.MIN_EPSILON = minEpsilon
        self.EPSILON_DECAY_RATE = decayRate
        self.BATCH_SIZE = batchSize
        self.TAU = tau

        self.currentNetwork = currNet
        self.targetNetwork  = targetNet
        self.steps = 0
        self.epsilon = 1

    def updateTargetModel(self):
        self.targetNetwork.setWeights(self.currentNetwork)

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.currentNetwork.outputSize - 1)
        else:
            return np.argmax(self.currentNetwork.predictOne(state))

    def observe(self, states, action, reward, newStates, done):
        self.replayBuffer.add( (states, action, reward, newStates, done) )
        self.steps += 1
        self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY_RATE)

        if self.steps % self.TAU == 0:
            self.updateTargetModel()

    def experienceReplay(self):
        # sample random batch from replay memory
        #if self.BATCH_SIZE > self.replayBuffer.size():
        if 1000 > self.replayBuffer.size():
            return
        batch = self.replayBuffer.sample(self.BATCH_SIZE)

        # combine all of the BATCH_SIZE states and new states into two lists
        # to let keras do the predictions in two calls
        states = np.array([x[0] for x in batch])
        newStates = np.array([x[3] for x in batch])
        currentPredictions = self.currentNetwork.predict(states)
        newPredictions = self.targetNetwork.predict(newStates)
        
        # Create list of xs and ys do train all of the batch items at once
        xs = np.zeros((self.BATCH_SIZE, self.currentNetwork.inputSize))
        ys = np.zeros((self.BATCH_SIZE, self.currentNetwork.outputSize))

        for i in range(self.BATCH_SIZE):
            state, action, reward, newState, done = batch[i]
            target = reward
            if not done:
                target += self.GAMMA * np.amax(newPredictions[i])

            # Approximately map current Q to the target Q
            currentQs = currentPredictions[i]
            currentQs[action] = target
            xs[i] = states[i] 
            ys[i] = currentQs

        # update network / fit the model
        self.currentNetwork.train(xs, ys)


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
replayBuffer = UniformReplayBuffer(2000)
currentNetwork = Network(0.001, env.stateSpaceSize, env.actionSpaceSize)
targetNetwork = Network(0.001, env.stateSpaceSize, env.actionSpaceSize)
agent = Agent(currentNetwork, targetNetwork, replayBuffer)

env.run(agent, 2500)
env.save("results/dqn_fixed_scores.txt", "results/dqn_fixed_scores.png")
