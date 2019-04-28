# -*- coding: utf-8 -*-

#This code makes an animation of Space Invaders from stored weights
#and outputs video files in RGB and in the preprocessed view which the agent sees

import gym
import numpy as np
import cv2
import CNN_agent as agent
import image_buffer
import time
import video_generator

#Resizes the image and outputs a binary color image
def preprocessing(observation,state_size):
    observation = cv2.resize(observation, state_size)
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255 , cv2.THRESH_BINARY)
    return np.reshape(observation, state_size+(1,))

if __name__ == "__main__":

    # Image buffer, enabling the agent to achieve a sense of time
    BUFFER_SIZE = 3
    img_buffer = image_buffer.Image_buffer(size=BUFFER_SIZE)

    # Preparing environment
    env = gym.make('SpaceInvaders-v0')
    state_size = (84, 84)
    action_size = env.action_space.n
    batch_size = 10
    agent = agent.Agent(state_size, action_size, BUFFER_SIZE)

    #Loading weights
    weight_file_name = "spaceinv_weights_e490.h5"
    try:
        agent.load("./results/CNN_SPACEINVADER_3_IMAGE_LONG_LINEAR/weights/"+weight_file_name)
    except:
        print("LOADING-ERROR")
        exit(1)

    #We don't want the agent to explore when we test its performance
    agent.epsilon = 0

    # Preparing for next run
    state = env.reset()
    state = preprocessing(state, state_size)
    done = False
    img_buffer.reset()
    img_buffer.append(state)

    # Video file generator
    output_video_filename = "test_vid"
    videoGenRGB = video_generator.VideoGenerator(24, 210, 160, 3)
    videoGenAgentView = video_generator.VideoGenerator(24, 84, 84, 1)

    while not done:

        env.render()

        # Agent performs an action
        action = agent.act(img_buffer.get_image_array())

        #Updating environment and score
        next_state, reward, done, _ = env.step(action)

        #Storing video
        videoGenRGB.append_frame(next_state)
        videoGenAgentView.append_frame(preprocessing(next_state, state_size))

        # Prepare next state
        next_state = state = preprocessing(next_state, state_size)

        # Append image buffer
        img_buffer.append(next_state)

    videoGenRGB.generate_video(output_video_filename+"-RGB",)
    videoGenAgentView.generate_video(output_video_filename+"-AGENT")
