#This code is made with the intention of giving the agent a sense of time

import numpy as np

#Image buffer class
class Image_buffer(object):

    def __init__(self, size):
        self.image_x = -1
        self.image_y = -1
        self.size = size
        self.buffer = []

    #Appends images to a buffer
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

    #Returns the buffer list as a buffer array
    def get_image_array(self):
        if len(self.buffer) == self.size:
            return np.array(self.buffer)#.reshape(1,self.image_x, self.image_y, self.size)
        else:
            print("BUFFER OTHER THAN EXPECTED")
            exit(1)


    def get_frame(self, index):
        if index >= self.size:
            print("FRAME INDEX OUT OF BOUNDS")
            exit(1)
        else:
            return np.array(self.buffer).reshape(1,self.image_x, self.image_y, index)

    #Resets the buffer
    def reset(self):
        self.buffer = []
