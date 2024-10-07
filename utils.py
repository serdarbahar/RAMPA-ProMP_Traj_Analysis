import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# these are needed to align orientations to x,y,z,w
def change_hand_trajectory(x):
    x[3], x[4], x[5], x[6] = x[4], x[6], x[3], x[5]
    return x[:,:4]
def chang_AR_trajectory(traj):
    return traj[:,:4]

def normalize_array(arr):
#Normalize an array between 0 and 1
#arr : all trajectories
    max_values = []
    min_values = []
    for traj in arr:
        traj = np.array(traj)
        max_values.append(np.max(traj))
        min_values.append(np.min(traj))

    max_value = np.max(max_values)
    min_value = np.min(min_values)
    norm_arr = []
    for traj in arr:
        traj = np.array(traj)
        traj = (traj-min_value) / (max_value-min_value)
        norm_arr.append(traj)

    return norm_arr

def load_trajectories(folder_path):
    if os.path.isdir(folder_path):
        hand_trajectories = []
        ar_trajectories = []

        hand_files = glob.glob(os.path.join(folder_path,"hand_trajectory*"))
        ar_files = glob.glob(os.path.join(folder_path,"ar_trajectory*"))

        for file in hand_files:
            traj = change_hand_trajectory(np.load(file))
            hand_trajectories.append(traj)

        for file in ar_files:
            traj = chang_AR_trajectory(np.load(file))
            ar_trajectories.append(traj)
        
        return hand_trajectories,ar_trajectories
    else:
        print("Folder not found")

def compute_jerk(traj, time_step):

    velocity = np.gradient(traj, time_step, axis=0)
    acceleration = np.gradient(velocity, time_step, axis=0)
    jerk = np.gradient(acceleration, time_step, axis=0)
        
    return jerk

def compute_parameters(traj ):

    mean_traj = np.mean(traj,axis=0)
    avg_deviation = np.mean(np.abs(traj - mean_traj))

    variation = np.sum(np.var(traj,axis=0))
    
    return avg_deviation, variation

def find_mean_std(array):
    return np.mean(array), np.std(array)

def plot_results(hand_parameters,ar_parameters):

    plt.figure(figsize=(14,6))

    plt.subplot(1,3,1)
    sns.boxplot([hand_parameters[:,0],ar_parameters[:,0]])
    plt.title("Jerk Comparision")
    plt.ylabel("Jerk")

    plt.subplot(1,3,2)
    sns.boxplot([hand_parameters[:,1],ar_parameters[:,1]])
    plt.title("Average Deviation Comparision")
    plt.ylabel("Average Deviation")

    plt.subplot(1,3,3)
    sns.boxplot([hand_parameters[:,2],ar_parameters[:,2]])
    plt.title("Variation Comparision")
    plt.ylabel("Variation")

    plt.tight_layout()
    plt.show()
