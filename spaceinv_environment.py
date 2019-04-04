import gym
import numpy as np
import cv2
import CNN_agent as agent

EPISODES = 1000

#Resizes the image to 84x84 and outputs a binary color image
def preprocessing(observation):
    observation = cv2.resize(observation, (84,84))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255 , cv2.THRESH_BINARY)

    #Slicing to remove unused screen for space invader, becomes 69,84
    observation = observation[10:79, 0:84]

    return np.reshape(observation, (69,84,1))

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    state_size = (69,84)
    action_size = env.action_space.n
    agent = agent.Agent(state_size, action_size)
    done = False
    batch_size = 10

    for e in range(EPISODES):
        state = env.reset()
        state = preprocessing(state)
        total_reward = 0
        for time in range(999999999):

            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if time % 20 == 0: print("Episode:", e,", Frame:", time, ", Total score:",total_reward, ", Action:", action, ", Memory size:", len(agent.memory), ", Epsilon:", agent.epsilon)

            next_state = state = preprocessing(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
