import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import random
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt

class DQNParameters:
    def __init__(self):
        self.learning_rate = 0.001
        self.gamma = .95
        self.min_epsilon = .01
        self.epsilon_decay_rate = .995
        self.num_episodes = 1000
        self.max_time_steps = 2000
        self.replayMemorySize = 2000
        self.batch_size = 32

class Game:
    def __init__(self, name):
        self.name = name
        self.ACTION_SIZE = 0
        self.STATE_SIZE  = 0

    def preProcessState(self, state):
        return state    

class PongGame(Game):
    def __init__(self):
        super().__init__('Pong-v0')
        self.ACTION_SIZE = 2
        self.STATE_SIZE  = 80 * 80

    def preProcessState(self, state):
        state = state[35:195] # crop
        state = state[::2,::2,0] # downsample by factor of 2
        state[state == 144] = 0 # erase background (background type 1)
        state[state == 109] = 0 # erase background (background type 2)
        state[state != 0] = 1 # everything else (paddles, ball) just set to 1
        return state.astype(np.float).ravel()
        
t= None
class DQN:
    def __init__(self, game, params=DQNParameters()):
        self.game = game
        self.env = gym.make(game.name)
        self.STATE_SIZE = game.STATE_SIZE
        self.ACTION_SIZE  = game.ACTION_SIZE
        self.GAMMA = params.gamma
        self.MIN_EPSILON = params.min_epsilon
        self.EPSILON_DECAY_RATE = params.epsilon_decay_rate
        self.NUM_EPISODES = params.num_episodes
        self.MAX_TIME_STEPS = params.max_time_steps
        self.replayMemory = deque(maxlen=params.replayMemorySize)
        self.BATCH_SIZE = params.batch_size
        self.LEARNING_RATE = params.learning_rate

        self.epsilon = 1.0
        self.model = self.buildModel(params.learning_rate)

    def buildModel(self, learning_rate):
        model = Sequential()
        model.add(Dense(256, input_dim=self.STATE_SIZE, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def selectAction(self, state):
        if random.random() < self.epsilon:
            #return self.env.action_space.sample()
            return random.randint(2, 3)
        else:
            predictedQ = model.predict(state)
            #return np.argmax(predicteQ[0])
            return np.argmax(predicteQ[0]) + 2

    def train(self, numEpisodes=None, render=False):
        if numEpisodes == None:
            numEpisodes = self.NUM_EPISODES
            
        for episode in range(numEpisodes):
            state = self.env.reset()
            state = self.game.preProcessState(state)
            state = state.reshape([1, self.STATE_SIZE])
            score = 0

            if self.MAX_TIME_STEPS == None:
                numTimeSteps = 10**6
            for timeStep in range(numTimeSteps):
                if render:
                    self.env.render()
                
                # select action
                action = self.selectAction(state)

                # execute that action
                new_state, reward, done, info = self.env.step(action)
                new_state = self.game.preProcessState(new_state)
                new_state = new_state.reshape([1, self.STATE_SIZE])
                
                # store experience in replay memory
                self.replayMemory.append((state, action, reward, new_state, done))

                # observe reward and next state
                score += reward
                state = new_state
                if done:
                    if (episode % 50) == 0:
                        print("Episode:", episode, " ended with score:", score)
                    break
                
            # sample random batch from replay memory
            if len(self.replayMemory) < self.BATCH_SIZE:
                continue
            batch = random.sample(self.replayMemory, self.BATCH_SIZE)
            
            # Preprocess states from batch
            states, targets = [], []
            for state, action, reward, new_state, done in batch:
                target = reward
                if not done:
                    target += self.GAMMA * np.max(self.model.predict(new_state))

                # Approximately map current Q to new Q
                currentQs = self.model.predict(state)
                currentQs[0][action - 2] = target

                states.append(state[0])
                targets.append(currentQs[0])

            global t
            t = states[0]
            # update network and epsilon
            history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
            loss = history.history['loss'][0]
            epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY_RATE)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)

    def getFname(self):
        return "dqn_pong-" + str(self.GAMMA) + str(self.MIN_EPSILON) + str(self.EPSILON_DECAY_RATE) + \
               str(self.LEARNING_RATE) + str(self.BATCH_SIZE) + ".txt"

        

    # Returns average score acheived with current network over 'games' episodes
    def evaluate(self, games=100):
        env = gym.make('Pong-v0')
        total_score = 0
        for episode in range(games):
            state = env.reset()
            state = self.game.preProcessState(state)
            state = np.reshape(state, [1, self.STATE_SIZE])
            score = 0
            done = False
            while not done:
                action = np.argmax(self.model.predict(state))

                new_state, reward, done, info = env.step(action)
                new_state = self.game.preProcessState(new_state)
                new_state = new_state.reshape([1, self.STATE_SIZE])
                
                score += reward
                state = new_state

            total_score += score
        return total_score / 100



game = PongGame()
#for lr in [0.0005, 0.001, 0.005]:
#    for gamma in [.9, .95, .99]:
#        for min_epsilon in [.001, .01, .1]:
#            for decay in [.9, .97, .995]:
#                for bs in [32, 64]:
params = DQNParameters()
params.learning_rate = 0.001
params.gamma = .95
params.max_time_steps = None
params.min_epsilon = .02
params.epsilon_decay_rate = .999
params.batch_size = 32

dqn = DQN(game, params)
dqn.train()

evals = []
for i in range(1, 1000):
    dqn.train(1000)
    #e = dqn.evaluate(100)
    #print("Evaluated score after", 100*i, "games:", e)
    #evals.append(e)
    dqn.save("tmpPongdqn.p")

final = dqn.evaluate(1000)
print(final)

