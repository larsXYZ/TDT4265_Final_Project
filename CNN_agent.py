import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from collections import deque
import time


class Agent(object):

    #Initialize the agent
    def __init__(self, number_of_states, number_of_actions, buffer_size):
        self.buffer_size = buffer_size
        self.state_size = number_of_states
        self.number_of_actions = number_of_actions
        self.learning_rate = 0.0001
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #Discount rate
        self.epsilon = 1.0 #Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.l2_const = 0.05
        self.model = self.create_model()

    #Create the model, the brain of the agent
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation='relu',
                                         input_shape=(self.state_size[0],self.state_size[1],self.buffer_size),
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_const),
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=int(time.time())),
                                         activity_regularizer=tf.keras.regularizers.l2(self.l2_const)))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_const),
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=int(time.time())),
                                         activity_regularizer=tf.keras.regularizers.l2(self.l2_const)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.l2_const),
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=int(time.time())),
                                        activity_regularizer=tf.keras.regularizers.l2(self.l2_const)))
        model.add(tf.keras.layers.Dense(units=self.number_of_actions, activation='linear',
                                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_const),
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=int(time.time())),
                                        activity_regularizer=tf.keras.regularizers.l2(self.l2_const)))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    #Store stuff for later learning
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Decide what to do depending on the current state
    def act(self, state_raw):

        #Reshapes the state array, keras expects it a certain way
        state = np.empty((1,self.state_size[0],self.state_size[1],self.buffer_size))
        for i in range(self.buffer_size):
            state[0, :, :, i] = state_raw[i, :, :, 0]

        if random.random() < self.epsilon: #Explore
            return random.randrange(self.number_of_actions)
        else: #Greedy
            action_scores = self.model.predict(x=state)
            best_action = np.argmax(action_scores)
            return best_action

    def replay(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        for sample in batch:

            #Reshapes the state array, keras expects it a certain way
            state_raw = sample[0]
            state = np.empty((1,self.state_size[0],self.state_size[1],self.buffer_size))
            for i in range(self.buffer_size):
                state[0, :, :, i] = state_raw[i, :, :, 0]

            #Reshapes the state array, keras expects it a certain way
            next_state_raw = sample[3]
            next_state = np.empty((1,self.state_size[0],self.state_size[1],self.buffer_size))
            for i in range(self.buffer_size):
                state[0, :, :, i] = state_raw[i, :, :, 0]
                
            action = sample[1]
            reward = sample[2]
            done = sample[4]
            
            #Check for diverged net
            q_vector = self.model.predict(next_state)
            for i in range(self.number_of_actions):
                if np.isnan(q_vector[0][i]):
                    print("Net has diverged")
                    exit(1)
                
                
            y = reward

            if not done:
                y = reward + self.gamma * np.amax(q_vector[0])

            y_pred = q_vector
            y_pred[0][action] = y

            self.model.fit(state, y_pred, epochs=1, verbose=0)

        #Reduces epsilon, making the agent more greedy
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self, filename):
        self.model.save_weights(filename)
