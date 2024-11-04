import numpy as np
from matplotlib import pyplot as plt
#  from continiously_point_dendrite import *
from point_dendrite import *
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
import time

data = np.load('P_bin_10.npy')[:38]
#  plt.plot(data)
#  plt.show()
#  exit()
alpha = np.arange(0,38,1)
fx = interp1d(alpha, data/data[0])




def restart_param():
    param = {'m1':0.219,
            'h1':0.0817,
            'n1':0.0008,
            'n2':0.00049,
            'n3':0.097,
            'm2':0.0474,
            'm3':0.0023,
            "Ca":1e-6,
            "Inmda":0,
            'n4':0.004427,
            'l1': 0.9989138754237681,
            'h0':0.3982,
            'm0':0.5412,
            'h00':0.3982,
            'm00':0.5412,
            'D':0.0,
            'weights':0,
            'N_syn':0,
            'k_max':0}
    E = {'K':-80,
         'K_c': -80,
         'Na': 56,
         'Cl': -68,
         'Ca':128}
    return param, E



E = {'K':-80, 'K_c': -80, 'Na': 56, 'Cl': -68, 'Ca':128}; # These values are frozen for now, will be changed later

#  E = {'K':-80, 'K_c': -70, 'Na': 56, 'Cl': -68, 'Ca':128}; # These values are frozen for now, will be changed later

angles = [0, 22, 45, 67, 90]
#  angles = [22]
KMAXs = [1, 2, 4, 6, 8, 10]
#  KMAXs = [6, 8, 10]
#  for k_max in tqdm.tqdm(KMAXs):
    #  for angle in angles:
        #  for i in range(30):

def angle_sorter(i):
    if i < 10:
        angle = 0
    elif i < 20:
        angle = 22.5
    elif i < 30:
        angle = 45
    elif i < 40:
        angle = 67.5
    elif i < 50:
        angle = 90
    return angle

def angle_sorter_close(i):
    if i < 10:
        angle = 0
    elif i < 20:
        angle = 5
    elif i < 30:
        angle = 10
    elif i < 40:
        angle = 15
    elif i < 50:
        angle = 20
    return angle



def sorter(i):
    if i < 10:
        idx = i%10
        k_max = 4
    elif i < 20:
        idx = i%10
        k_max = 5
    elif i < 30:
        idx = i%10
        k_max = 6
    elif i < 40:
        idx = i%10
        k_max = 8
    elif i < 50:
        idx = i%10
        k_max = 10
    elif i < 60:
        idx = i%10
        k_max = 12
    elif i < 70:
        idx = i%10
        k_max = 14
    elif i < 80:
        idx = i%10
        k_max = 16
    elif i < 90:
        idx = i%10
        k_max = 18
    return idx, k_max

def runi(i):
    angle = 0
    idx, k_max = sorter(i)
    delays, weights, N_syn, weights2 = create_weight_and_delay('clustered', angle, i)
    param, E = restart_param()
    param['weights'] = weights
    param['weights2'] = weights2*0
    param['N_syn'] = N_syn
    param['k_max'] = k_max
    E['K_c'] += k_max
    data_0 = simulation(param, E, delays, False, change = True)
    np.save(f'EK_effect/no_noise_clamp_data_{idx}_clu_{angle}_{k_max}', data_0)

def run2(i):
    param, E = restart_param()
    delays = np.arange(4)
    data_0 = simulation(param, E, delays, i, False, change = True)
    #  V = data_0.item().get('V')[::100]
    V = data_0['V']
    #  RT = spike.item().get('RT')[::100]
    #  np.save(f'cont/spike_{i}', V[::100])
    np.save(f'cont/80weight_{i}', V[::100])


def run(i):
    angle = 0
    #  idx, k_max = sorter(i)
    angle = angle_sorter_close(i)
    #  if angle < 38:
    k_max = 10*fx(angle)
    #  else:
        #  k_max = 0
    #  k_max = 0
    idx = i%10
    #  angle += np.random.randn()*0.01
    delays, weights, N_syn, weights2 = create_weight_and_delay('clustered', angle, i)
    #  delays, weights, N_syn, weights2 = create_spike_and_delay('clustered', angle, i)
    param, E = restart_param()
    param['weights'] = weights*0.85
    param['weights2'] = weights2*0.85*0.5
    param['N_syn'] = N_syn
    param['k_max'] = k_max
    E['K_c'] += k_max
    data_0 = simulation(param, E, delays, False, change = True)
    V = data_0['V']
    np.save(f'single_traces_fall/data_{idx}_clu_{angle}_GABA_TRUE', V[::100])


#  run2(0)
if __name__ == '__main__':
    index = np.arange(30, 40,1)
    #  pool = Pool(cpu_count() - 2)
    pool = Pool(5)
    pool.map(run, index)
    pool.close()
    
    pool.join()
