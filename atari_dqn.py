import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.initializers import VarianceScaling
import random
import numpy as np
import math
import gym
from collections import deque
import matplotlib.pyplot as plt
import time

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

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

    def __len__(self):
        return self.size()

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
        samples = random.sample(range(FRAMES_SKIPPED, len(self.memory) - 1), size)
        batch = []
        for i in range(size):
            frame = samples[i]
            states = []
            for f in range(frame - 3, frame + 1):
                states.append(self.memory[f][0])
            newStates = []
            for f in range(frame - 2, frame + 2):
                newStates.append(self.memory[f][0])
            _, a, r, d = self.memory[frame]
            batch.append((states, a, r, newStates, d))

        return batch

HUBER_LOSS_DELTA = 1.0
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)

    return K.mean(loss)

class Network:
    def __init__(self, learningRate, inputShape, outputSize):
        self.learningRate = learningRate
        self.inputShape = inputShape
        self.outputSize= outputSize
        self.model = self.buildModel()

    def buildModel(self):
        model = Sequential()
        #model.add(Lambda(lambda x: x / 255.0, output_shape=self.inputShape))
        model.add(Conv2D(16, (8, 8), strides=(4,4), activation='relu', input_shape=(self.inputShape), data_format='channels_first',
                         kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Conv2D(32, (4, 4), strides=(2,2), activation='relu', kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu', kernel_initializer=VarianceScaling(scale=2.0)))
        model.add(Dense(units=self.outputSize, activation='linear', kernel_initializer=VarianceScaling(scale=2.0)))
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learningRate))
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
    def __init__(self, currentNet, tNetwork, buffer, gamma = 0.99, minEpsilon = 0.1, explorationFrames = 1000000, batchSize = 32, tau = 10000):
        self.replayBuffer = buffer
        self.GAMMA = gamma
        self.MIN_EPSILON = minEpsilon
        self.EXPLORATION_FRAMES = explorationFrames
        self.BATCH_SIZE = batchSize

        self.currentNetwork = currentNet
        self.targetNetwork = tNetwork
        self.updateTargetModel()
        self.TAU = tau
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
        self.replayBuffer.add(state, action, reward, newState, done)
        self.steps += 1
        if self.steps <= self.EXPLORATION_FRAMES:
            self.epsilon = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * (self.EXPLORATION_FRAMES - self.steps) / self.EXPLORATION_FRAMES

        if self.steps % self.TAU == 0:
            self.updateTargetModel()

    def experienceReplay(self):
        # sample random batch from replay memory
        batch = self.replayBuffer.sample(self.BATCH_SIZE)

        # combine all of the BATCH_SIZE states and new states into two lists
        # to let keras do the predictions in two calls
        states = np.array([x[0] for x in batch]).astype(np.float32) / 255.0
        newStates = np.array([x[3] for x in batch]).astype(np.float32) / 255.0
        
        currentPredictions = self.currentNetwork.predict(states)
        newPredictions = self.targetNetwork.predict(newStates)
        
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

stateStack = None
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
        reward = np.clip(reward, -1, 1)
        newState = self.processState(newState)
        return newState, reward, done, info

    def run(self, agent, numSteps, verbose = True, render = False):
        global stateStack
        self.agent = agent
        episode = 0
        step = 0
        while step < numSteps:
            state = self.env.reset()
            for _ in range(random.randint(1, 30)):
                state , _, _, _ = self.env.step(0)
            score, lives = 0, 5
            state = self.processState(state)
            done = False
            dead = False
            stateStack = np.array([state, state, state, state]).astype(np.float32) / 255.0
            
            while not done:
                if render:
                    self.env.render()
                    time.sleep(.3)
                
                # select action
                action = agent.selectAction(stateStack)

                # execute that action
                newState, reward, done, info = self.stepEnvironment(action)
                if lives > info['ale.lives']:
                    dead = True
                    lives = info['ale.lives']
                
                # store experience in replay memory
                agent.observe(state, action, reward, newState, dead) # use dead, not done for breakout
                agent.experienceReplay()

                # observe reward and next state
                score += reward
                state = newState
                s = newState.astype(np.float32) / 255.0
                s = self.stateSpaceShape
                stateStack = np.append(stateStack[1:], state.reshape((1, s[1], s[2])), axis=0)

                if dead:
                    dead = False
                step += 1

            episode += 1
            self.scores.append(score)
            if verbose:
                print("Episode:", episode, ", step:", agent.steps, ":", score)
                if episode % 100 == 0:
                    print("Last 100 episode average:", sum(self.scores[-100:]) / 100)
                if agent.steps % 1000:
                    agent.currentNetwork.save("breakout_cnet.h5")
                    agent.targetNetwork.save("breakout_tnet.h5")

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
replayBuffer = UniformReplayBuffer(600000)

randomAgent = RandomAgent(replayBuffer, env.actionSpaceSize)
REPLAY_BUFFER_START_SIZE = 50000
while replayBuffer.size() < REPLAY_BUFFER_START_SIZE:
    env.run(randomAgent, 1000, False, False)
    print(replayBuffer.size() / REPLAY_BUFFER_START_SIZE)
print("Replay Buffer Initialized")

cNetwork = Network(0.00001, env.stateSpaceShape, env.actionSpaceSize)
tNetwork = Network(0.00001, env.stateSpaceShape, env.actionSpaceSize)
agent = Agent(cNetwork, tNetwork, replayBuffer)

env.run(agent, 10000000)
env.fullSave('breakout_dqn')
