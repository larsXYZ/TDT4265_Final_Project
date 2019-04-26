import random
import numpy as np
import tensorflow as tf
from collections import deque

class Agent(object):

    #Initialize the agent
    def __init__(self, number_of_states, number_of_actions):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.learning_rate = 0.001
        self.memory = deque(maxlen=200000)
        self.gamma = 0.95 #Discount rate
        self.epsilon = 1.0 #Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.model = self.create_model()

    #Create the model, the brain of the agent
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=30, input_dim=self.number_of_states, activation='relu'))
        model.add(tf.keras.layers.Dense(units=30, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.number_of_actions, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        return model

    #Store stuff for later learning
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Decide what to do depending on the current state
    def act(self, state):
        if random.random() < self.epsilon: #Explore
            return random.randrange(self.number_of_actions)
        else: #Greedy
            return np.argmax(self.model.predict(x=state))

    def replay(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        for sample in batch:

            state = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state = sample[3]
            done = sample[4]

            y = reward

            if not done:
                y = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            y_pred = self.model.predict(state)
            y_pred[0][action] = y

            self.model.fit(state, y_pred, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)
