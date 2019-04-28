import gym
import numpy as np
import cv2
import CNN_agent as agent
import pickle
import sys
import matplotlib.pyplot as plt
import best_agent_tracker
import image_buffer

#Resizes the image and outputs a binary color image
def preprocessing(observation,state_size):
    observation = cv2.resize(observation, state_size)
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255 , cv2.THRESH_BINARY)
    return np.reshape(observation, state_size+(1,))

#Stores all relevant data to continue training
def autosave(agent, score_storage, e):
    agent.save("./autosave/spaceinv_weights.h5")
    np.save("./autosave/spaceinv_score_storage", score_storage)
    pickle.dump(agent.memory, open("./autosave/spaceinv_memory.p", "wb"))
    pickle.dump(e, open("./autosave/spaceinv_episode_count.p", "wb"))
    print("Autosave")

#Loads all relevant data to continue training
def autoload(agent):
    agent.load("./autosave/spaceinv_weights.h5")
    score_storage = np.copy(np.load("./autosave/spaceinv_score_storage.npy"))
    agent.memory = pickle.load(open("./autosave/spaceinv_memory.p", 'rb'))
    e = pickle.load(open("./autosave/spaceinv_episode_count.p", "rb"))
    print("Autoload, started at episode:", e)
    return e, score_storage

EPISODES = 500

if __name__ == "__main__":

    try:
        save = input("Do you want to save your weights? (y/n)\n")
        load = input("Do you want to load saved weights and memory? (y/n)\n")

        #Score storage
        score_storage = np.zeros(EPISODES)

        #Keeps track of best agent
        tracker = best_agent_tracker.Best_Agent_tracker()

        #Image buffer, enabling the agent to achieve a sense of time
        BUFFER_SIZE = 4
        img_buffer = image_buffer.Image_buffer(size=BUFFER_SIZE)

        #Preparing environment
        env = gym.make('SpaceInvaders-v0')
        state_size = (84,84)
        action_size = env.action_space.n
        batch_size = 10
        agent = agent.Agent(state_size, action_size, BUFFER_SIZE)

        #Loading autosave
        e = 0
        if load == "y":
            e, score_storage = autoload(agent)

        while e < EPISODES:

            #Preparing for next run
            state = env.reset()
            state = preprocessing(state,state_size)
            done = False
            img_buffer.reset()
            img_buffer.append(state)

            total_reward = 0
            for time in range(999999999):

                #env.render()



                #Agent performs an action
                action = agent.act(img_buffer.get_image_array())

                #The environment is stepped forward
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                #Prepare next state
                next_state = state = preprocessing(next_state,state_size)

                #Recording for memories
                state_previous = img_buffer.get_image_array()
                img_buffer.append(next_state)
                state_resulting = img_buffer.get_image_array()

                #Agent remembers
                agent.remember(state_previous, action, reward, state_resulting, done)

                #Update next state
                state = next_state

                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, total_reward, agent.epsilon))
                    score_storage[e] = total_reward
                    tracker.update_best_agent(agent, total_reward) # Checks if this agent is best agent yet
                    break


                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            if save == 'y' and e % 10 == 0:
                agent.save("./weights/spaceinv_weights_e" + str(e) + ".h5" )
                tracker.get_best_agent().save("./weights/spaceinv_weights_best.h5")
                autosave(agent, score_storage, e)

            e += 1


    #Saves data before exiting
    except:
        print("EXCEPTION")
        if save == 'y':
            tracker.get_best_agent().save("./weights/spaceinv_weights_best.h5")
            autosave(agent, score_storage, e)
        sys.exit()


    #Saving and plotting result
    agent.save("./weights/spaceinv_weights_final.h5")
    tracker.get_best_agent().save("./weights/spaceinv_weights_best.h5")
    autosave(agent, score_storage, e)
    plt.plot( np.arange(1,EPISODES+1),score_storage)
    plt.savefig("cnn_agent_plot")
