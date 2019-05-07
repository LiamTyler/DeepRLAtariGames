import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import random
import numpy as np
import math
import gym
from collections import deque
import matplotlib.pyplot as plt
import time


ATARI_SHAPE = (4, 105, 80)
FRAMES_SKIPPED = 4

# RingBuffer from https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
class RingBuffer:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __setitem__(self, idx, value):
            self.data[idx] = value
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class UniformReplayBuffer:
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.memory = RingBuffer(capacity)

    def size(self):
        return len(self.memory)

    # store as (state, action, reward, done)
    def add(self, state, action, reward, newState, done):
        L = len(self.memory)
        if L == 0:
            self.memory.append((state, action, reward, done))
            self.memory.append((newState, action, reward, done))
        else:
            self.memory[(self.memory.end - 1) % len(self.memory.data)] = (state, action, reward, done)
            self.memory.append((newState, action, reward, done))


    def isFull(self):
        return self.size() == self.capacity

    def sample(self, size):
        samples = random.sample(range(FRAMES_SKIPPED, len(self.memory)), size)
        batch = []
        for i in range(size):
            frame = samples[i]
            states = []
            for f in range(frame - 3, frame + 1):
                states.append(self.memory[frame][0])
            newStates = []
            for f in range(frame - 2, frame + 2):
                newStates.append(self.memory[frame][0])
            _, a, r, d = self.memory[frame]
            batch.append((states, a, r, newStates, d))

        return batch

class Network:
    def __init__(self, learningRate, inputShape, outputSize):
        self.learningRate = learningRate
        self.inputShape = inputShape
        self.outputSize= outputSize
        self.model = self.buildModel()

    def buildModel(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0, output_shape=self.inputShape))
        model.add(Conv2D(16, (8, 8), strides=(4,4), activation='relu', input_shape=(self.inputShape), data_format='channels_first'))
        model.add(Conv2D(32, (4, 4), strides=(2,2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.outputSize, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learningRate, rho=0.95, epsilon=0.01))
        return model

    def predict(self, xs):
        return self.model.predict(xs)

    def predictOne(self, x):
        s = self.inputShape
        return self.model.predict(x.reshape(1, s[0], s[1], s[2])).flatten()
    
    def train(self, xs, ys):
        self.model.fit(xs, ys, epochs=1, verbose=0)

    def setWeights(self, otherNetwork):
        self.model.set_weights(otherNetwork.model.get_weights())

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

class Agent:
    def __init__(self, currentNet, buffer, gamma = 0.99, minEpsilon = 0.1, explorationFrames = 1000000, batchSize = 32):
        self.replayBuffer = buffer
        self.GAMMA = gamma
        self.MIN_EPSILON = minEpsilon
        self.EXPLORATION_FRAMES = explorationFrames
        self.BATCH_SIZE = batchSize

        self.currentNetwork = currentNet
        self.steps = 0
        self.epsilon = 1

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.currentNetwork.outputSize - 1)
        else:
            return np.argmax(self.currentNetwork.predictOne(state))

    def observe(self, state, action, reward, newState, done):
        self.replayBuffer.add(state, action, reward, newState, done)
        self.steps += 1
        if self.steps <= self.EXPLORATION_FRAMES:
            self.epsilon = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * (self.EXPLORATION_FRAMES - self.steps) / self.EXPLORATION_FRAMES

    def experienceReplay(self):
        # sample random batch from replay memory
        if self.BATCH_SIZE + 4 > self.replayBuffer.size():
            return
        batch = self.replayBuffer.sample(self.BATCH_SIZE)

        # combine all of the BATCH_SIZE states and new states into two lists
        # to let keras do the predictions in two calls
        states = np.array([x[0] for x in batch])
        newStates = np.array([x[3] for x in batch])
        
        currentPredictions = self.currentNetwork.predict(states)
        newPredictions = self.currentNetwork.predict(newStates)
        
        # Create list of xs and ys do train all of the batch items at once
        s = self.currentNetwork.inputShape
        xs = np.zeros((self.BATCH_SIZE, s[0], s[1], s[2]))
        #ys = np.zeros((self.BATCH_SIZE, self.currentNetwork.outputSize))

        for i in range(self.BATCH_SIZE):
            _, action, reward, _, done = batch[i]
            target = reward
            if not done:
                #target += self.GAMMA * newPredictions[i][np.argmax(newPredictions[i])]
                target += self.GAMMA * np.amax(newPredictions[i])

            # Approximately map current Q to the target Q
            currentPredictions[i][action] = target
            xs[i] = states[i] 
            #ys[i] = currentQs

        # update network / fit the model
        self.currentNetwork.train(xs, currentPredictions)

class RandomAgent:
    def __init__(self, buffer, actionSpace):
        self.replayBuffer = buffer
        self.steps = 0
        self.actionSpace = actionSpace

    def selectAction(self, state):
        return random.randint(0, self.actionSpace - 1)

    def observe(self, state, action, reward, newState, done):
        self.replayBuffer.add(state, action, reward, newState, done)
        self.steps += 1

    def experienceReplay(self):
        pass


class Environment:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(name)
        self.stateSpaceShape = ATARI_SHAPE
        self.actionSpaceSize = self.env.action_space.n
        self.scores = []
        self.agent = None

    def saveScores(self, scoresPath, figPath):
        s = str(self.scores)
        s = s[1:-1]
        f = open(scoresPath, "w")
        f.write(s)
        f.close()
        plt.plot(list(range(len(self.scores)))[::2], self.scores[::2])
        plt.savefig(figPath)

    def fullSave(self, name):
        f = open(name + '.txt', 'w')
        f.write(self.name + 'n') # env name
        f.write(str(len(self.scores)) + '\n') # number of episodes
        f.write(str(self.scores)[1:-1] + '\n') # scores
        if agent:
            f.write(str(self.agent.steps) + '\n') # steps
            self.agent.model.save(name + '.h5')

    def plot(self):
        plt.plot(list(range(len(self.scores)))[::2], self.scores[::2])
        plt.show()

    def reset(self):
        s = self.env.reset()
        return self.processState(s)

    def processState(self, I):
        I = I[::2, ::2]
        return np.mean(I, axis=2).astype(np.uint8)

    def stepEnvironment(self, action):
        newState, reward, done, info = self.env.step(action)
        newState = self.processState(newState)
        return newState, reward, done, info

    def run(self, agent, numEpisodes, verbose = True, render = False):
        self.agent = agent
        for episode in range(numEpisodes):
            state = self.processState(self.env.reset())
            score = 0            
            done = False
            stateStack = np.array([state, state, state, state])
            
            while not done:
                if render:
                    self.env.render()
                
                # select action
                action = agent.selectAction(stateStack)

                # execute that action
                newState, reward, done, info = self.stepEnvironment(action)
                
                # store experience in replay memory
                agent.observe(state, action, reward, newState, done)
                agent.experienceReplay()

                # observe reward and next state
                score += reward
                state = newState
                s = self.stateSpaceShape
                stateStack = np.append(stateStack[1:], state.reshape((1, s[1], s[2])), axis=0)

            self.scores.append(score)
            if verbose:
                print("Episode", episode, ":", score)

def run(env):
    for episode in range(10):
        state = env.reset()
        score = 0  
        done = False
        
        while not done:
            env.render()
            time.sleep(.3)
            
            # select action
            action = env.action_space.sample()

            # execute that action
            newState, reward, done, info = env.step(action)

            state = newState
            
env = Environment('BreakoutDeterministic-v4')
replayBuffer = UniformReplayBuffer(1000000)

randomAgent = RandomAgent(replayBuffer, env.actionSpaceSize)
REPLAY_BUFFER_START_SIZE = 5000
while replayBuffer.size() < REPLAY_BUFFER_START_SIZE:
    env.run(randomAgent, 1, False, False)
    print(replayBuffer.size() / REPLAY_BUFFER_START_SIZE)
print("full")

cNetwork = Network(0.00025, env.stateSpaceShape, env.actionSpaceSize)
agent = Agent(cNetwork, replayBuffer)

env.run(agent, 10000)
env.fullSave('breakout_dqn')
