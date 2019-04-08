import gym
import numpy as np
import cv2
import CNN_agent as agent
import sys
#import matplotlib.pyplot as plt

#Image buffer class
class Image_buffer(object):
    def __init__(self, size):
        self.image_x = -1
        self.image_y = -1
        self.size = size
        self.buffer = []

    def append(self, image):
        if len(self.buffer) == self.size:
            self.buffer.pop(0)
            self.append(image)
        elif len(self.buffer) < self.size:
            while len(self.buffer) < self.size:
                self.image_x = np.shape(image)[0]
                self.image_y = np.shape(image)[1]
                self.buffer.append(image)
        else:
            print("BUFFER OTHER THAN EXPECTED")
            exit(1)

    def get_image_array(self):
        if len(self.buffer) == self.size:
            return np.array(self.buffer).reshape(1,self.image_x, self.image_y, self.size)
        else:
            print("BUFFER OTHER THAN EXPECTED")
            exit(1)

EPISODES = 1000

#Resizes the image to 84x84 and outputs a binary color image
def preprocessing(observation):
    observation = cv2.resize(observation, (84,84))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255 , cv2.THRESH_BINARY)

    return np.reshape(observation, (84,84,1))

if __name__ == "__main__":

    try:
        save = input("Do you want to save your weights (y/n)\n")

        #Score storage
        score_storage = np.zeros(EPISODES)

        #Image buffer, enabling the agent to achieve a sense of time
        BUFFER_SIZE = 3
        image_buffer = Image_buffer(size=BUFFER_SIZE)

        env = gym.make('SpaceInvaders-v0')
        state_size = (84,84)
        action_size = env.action_space.n
        agent = agent.Agent(state_size, action_size, BUFFER_SIZE)
        #agent.load("./weights/spaceinv_weights.h5")
        done = False
        batch_size = 10

        for e in range(EPISODES):
            state = env.reset()
            state = preprocessing(state)
            image_buffer.append(state)

            total_reward = 0
            for time in range(999999999):

                #env.render()

                action = agent.act(image_buffer.get_image_array())
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                if time % 20 == 0: print("Episode:", e,", Frame:", time, ", Total score:",total_reward, ", Action:", action, ", Memory size:", len(agent.memory), ", Epsilon:", agent.epsilon)

                next_state = state = preprocessing(next_state)

                #Recording for memories
                state_old = image_buffer.get_image_array()
                image_buffer.append(next_state)
                state_new = image_buffer.get_image_array()

                agent.remember(state_old, action, reward, state_new, done)

                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, total_reward, agent.epsilon))
                    score_storage[e] = total_reward
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            if save == 'y' and e % 20 == 0:
                agent.save("./weights/spaceinv_weights_e" + str(e) + '.h5' )
    except KeyboardInterrupt:
        if save == 'y':
            agent.save("./weights/spaceinv_weights.h5")
            np.save("score_storage", score_storage)
            print("saved file")
        sys.exit()


    #Saving and plotting result
    agent.save("./weights/spaceinv_weights.h5")
    np.save("score_storage", score_storage)
    plt.plot( np.arange(1,EPISODES+1),score_storage)
    plt.show()
