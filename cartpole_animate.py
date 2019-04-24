# -*- coding: utf-8 -*-
import gym
import numpy as np
import fully_connected_agent as agent
import matplotlib.pyplot as plt
import pickle
import time

if __name__ == "__main__":

    #Defines environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    done = False

    #Defines agent
    agent = agent.Agent(state_size, action_size)
    batch_size = 32

    #Loading weights
    weight_file_name = "cartpole_weights_best.h5"
    try:
        agent.load("./animate/"+weight_file_name)
    except:
        print("LOADING-ERROR")
        exit(1)

    #Prepares for run
    state = env.reset()

    while not done:

        env.render()

        #Agents performs action
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)

        #Updating environment and score
        next_state, reward, done, _ = env.step(action)

        #Prepares for next state
        state = next_state

        #Wait some time
        time.sleep(0.1)
