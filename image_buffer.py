import numpy as np

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

    def reset(self):
        self.buffer = []
