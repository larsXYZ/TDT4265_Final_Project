import gym
import numpy as np
import cv2
import tensorflow as tf
#import matplotlib.pyplot as plt
from collections import deque
#import fully_connected_agent as agent
import CNN_agent as agent

EPISODES = 1000

#Resizes the image to 84x84 and outputs a binary color image
def preprocessing(observation):
    observation = cv2.resize(observation, (84,84))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255 , cv2.THRESH_BINARY)
    return np.reshape(observation, (84,84,1))

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    print(env.observation_space)
    state_size = (84,84,1)
    print(state_size)
    action_size = env.action_space.n
    print(action_size)
    agent = agent.Agent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = preprocessing(state)
        for time in range(500):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #print(time, reward)

            reward = reward if not done else -10
            next_state = state = preprocessing(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
