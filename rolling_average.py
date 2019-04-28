import numpy as np
import pickle
import matplotlib.pyplot as plt

#Function implementing rolling average, for more readable plots
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#Retrieving data
score_storage = np.copy(np.load("spaceinv_score_storage.npy"))
EPISODES = np.shape(score_storage)[0]

#Computing moving average
score_storage_roll_av = moving_average(score_storage,n=20)

#Plotting
plt.plot(np.arange(1, EPISODES + 1), score_storage)
plt.show()

plt.plot(np.arange(1, EPISODES + 2 - n), score_storage_roll_av)
plt.show()
