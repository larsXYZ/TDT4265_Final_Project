import random
import numpy as np
import tensorflow as tf
from collections import deque


class Agent(object):

    #Initialize the agent
    def __init__(self, number_of_states, number_of_actions, buffer_size):
        self.buffer_size = buffer_size
        self.state_size = number_of_states
        self.number_of_actions = number_of_actions
        self.learning_rate = 0.001
        self.memory = deque()
        self.gamma = 0.95 #Discount rate
        self.epsilon = 1.0 #Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.model = self.create_model()

    #Create the model, the brain of the agent
    def create_model(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=(self.state_size[0],self.state_size[1],self.buffer_size)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.number_of_actions, activation='relu'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        return model

    #Store stuff for later learning
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Decide what to do depending on the current state
    def act(self, state):

        state = state.reshape(1,self.state_size[0],self.state_size[1],self.buffer_size)

        if random.random() < self.epsilon: #Explore
            return random.randrange(self.number_of_actions)
        else: #Greedy
            action_scores = self.model.predict(x=state)
            best_action = np.argmax(action_scores)
            return best_action

    def replay(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        for sample in batch:

            state = (sample[0]).reshape(1,self.state_size[0],self.state_size[1],self.buffer_size)
            action = sample[1]
            reward = sample[2]
            next_state = (sample[3]).reshape(1,self.state_size[0],self.state_size[1],self.buffer_size)
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
