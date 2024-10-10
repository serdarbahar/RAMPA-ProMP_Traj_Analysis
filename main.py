import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
import utils 
import seaborn as sns
from scipy import signal


base_path = "./Rampa_Trajectories"  #FIX ACCORDINGLY
time_step_1 = 1.0  # 1s record rate
time_step_2 = 0.25

hand_parameters = []
ar_parameters = []

for folder_name in os.listdir(base_path):
    if folder_name == ".DS_Store": #Ignore
        continue
    else:
        folder_path = os.path.join(base_path,folder_name)
        hand_trajectories, ar_trajectories = utils.load_trajectories(folder_path)

        hand_trajectories = utils.normalize_array(hand_trajectories)
        ar_trajectories = utils.normalize_array(ar_trajectories)

        jerk_hand_traj = []
        jerk_ar_traj = []
        for i in range(3):
            position_1 = np.array(hand_trajectories[i])
            position_2 = np.array(ar_trajectories[i])

            #This is needed since record rate for ar trajectories are difference
            num_samples = len(position_2)
            position_2_resampled = signal.resample(position_2, num_samples) 

            jerk_h = utils.compute_jerk(position_1, time_step_1)
            jerk_ar = utils.compute_jerk(position_2_resampled, time_step_1)
               
            # Compare smoothness
            jerk_rms_1 = np.sqrt(np.mean(jerk_h**2))
            jerk_rms_2 = np.sqrt(np.mean(jerk_ar**2))

            jerk_hand_traj = np.append(jerk_hand_traj,jerk_rms_1)
            jerk_ar_traj = np.append(jerk_ar_traj,jerk_rms_2)
                

        j_h= np.mean(np.array(jerk_hand_traj))
        j_ar = np.mean(np.array(jerk_ar_traj))   
                                  
        for traj in hand_trajectories:
            d_h,v_h = utils.compute_parameters(traj)
        hand_parameters.append([j_h,d_h,v_h])

        for traj in ar_trajectories:
            d_ar,v_ar = utils.compute_parameters(traj)
        ar_parameters.append([j_ar,d_ar,v_ar])
    t1,t2,t3,t4 = d_h,v_h,d_ar,v_ar
    total_error = t1+t2+t3+t4
    hand = t1+t2
    ar = t3+t4
    print(f"{folder_name} total: {total_error:.3f} ar : {ar:.3f}")
            

hand_parameters = np.array(hand_parameters)
ar_parameters = np.array(ar_parameters)

hand_jerk_mean,hand_jerk_std = utils.find_mean_std(hand_parameters[:,0])
ar_jerk_mean,ar_jerk_std = utils.find_mean_std(ar_parameters[:,0])

hand_avg_deviation_mean,hand_avg_deviation_std = utils.find_mean_std(hand_parameters[:,1])
ar_avg_deviation_mean,ar_avg_deviation_std = utils.find_mean_std(ar_parameters[:,1])

hand_variation_mean,hand_variation_std = utils.find_mean_std(hand_parameters[:,2])
ar_variation_mean,ar_variation_std = utils.find_mean_std(ar_parameters[:,2])

print(f"Hand Jerk Mean: {hand_jerk_mean:.3f} Hand Jerk Std: {hand_jerk_std:.3f}")
print(f"AR Jerk Mean: {ar_jerk_mean:.3f} AR Jerk Std: {ar_jerk_std:.3f}")
print()
print(f"Hand Avg Deviation Mean: {hand_avg_deviation_mean:.3f} Hand Avg Deviation Std: {hand_avg_deviation_std:.3f}")
print(f"AR Avg Deviation Mean: {ar_avg_deviation_mean:.3f} AR Avg Deviation Std: {ar_avg_deviation_std:.3f}")
print()
print(f"Hand Variation Mean: {hand_variation_mean:.3f} Hand Variation Std: {hand_variation_std:.3f}")
print(f"AR Variation Mean: {ar_variation_mean:.3f} AR Variation Std: {ar_variation_std:.3f}")

utils.plot_results(np.array(hand_parameters),np.array(ar_parameters))
