# -*- coding: utf-8 -*-
import gym
import numpy as np
import fully_connected_agent as agent
import matplotlib.pyplot as plt
import pickle
import best_agent_tracker

#SOURCE https://keon.io/deep-q-learning/
EPISODES = 100

if __name__ == "__main__":

    #Do we save the weights?
    save = input("Do you want to save your weights and score? (y/n)\n")

    #Defines environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    done = False

    #Defines agent
    agent = agent.Agent(state_size, action_size)
    batch_size = 32

    #Defines best agent tracker
    tracker = best_agent_tracker.Best_Agent_tracker()

    # Score storage
    score_storage = np.zeros(EPISODES)

    for e in range(EPISODES):

        #Prepares for next run
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(1000):

            #env.render()

            #Agents performs action
            action = agent.act(state)

            #Updating environment and score
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10

            #The agent will remember this
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)

            #Prepares for next state
            state = next_state

            #Printing result of the episode
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                score_storage[e] = time

                #Saving weights
                if save == 'y' and e % 20 == 0:
                    agent.save("./weights/cartpole_weights_e" + str(e) + '.h5')

                # Checks if this agent is best agent yet
                tracker.update_best_agent(agent, time)

                break

            #The agent learns from memories
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    #Creates plot and saves agent
    if save == 'y':
        agent.save("./weights/cartpole_weights_final.h5")
        np.save("cartpole_score_storage", score_storage)
        tracker.get_best_agent().save("./weights/cartpole_weights_best.h5")

    plt.plot( np.arange(1,EPISODES+1),score_storage)
    plt.show()
