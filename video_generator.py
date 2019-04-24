from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np

#Generates video from a source array

class VideoGenerator(object):

    def __init__(self, FPS, width, height, color_depth):

        self.FPS = float(FPS)
        self.width = int(width)
        self.height = int(height)
        self.color_depth = int(color_depth)
        self.frame_array = []

    def append_frame(self, frame):

        assert np.shape(frame) == (self.width, self.height, self.color_depth)
        self.frame_array.append(frame)



    def generate_video(self, filename):

        fourcc = VideoWriter_fourcc(*'DIVX')
        video = VideoWriter("./"+filename + ".avi", fourcc, self.FPS, (self.height, self.width))

        n_frames = len(self.frame_array)
        for i in range(n_frames):

            frame = np.array(self.frame_array[i])

            #Swapping Red and Blue channels or cloning them
            if self.color_depth > 1:
                temp = np.copy(frame[:,:,0])
                frame[:,:,0] = frame[:,:,2]
                frame[:, :, 2] = temp
            else:
                frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)

            frame = np.squeeze(frame)

            video.write(frame)

        video.release()