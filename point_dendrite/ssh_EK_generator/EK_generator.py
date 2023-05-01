import numpy as np
from matplotlib import pyplot as plt
from point_dendrite import *
from multiprocessing import Pool, cpu_count


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

E = {'K':-80, 'K_c': -70, 'Na': 56, 'Cl': -68, 'Ca':128}; # These values are frozen for now, will be changed later

angles = [0, 22, 45, 67, 90]
#  angles = [22]
KMAXs = [1, 2, 4, 6, 8, 10]
#  KMAXs = [6, 8, 10]
#  for k_max in tqdm.tqdm(KMAXs):
    #  for angle in angles:
        #  for i in range(30):
def sorter(i):
    if i < 10:
        idx = i%20
        k_max = 4
    elif i < 20:
        idx = i%20
        k_max = 5
    elif i < 30:
        idx = i%20
        k_max = 6
    elif i < 40:
        idx = i%20
        k_max = 8
    elif i < 50:
        idx = i%20
        k_max = 10
    return idx, k_max

def run(i):
    print(i)
    angle = 0
    idx, k_max = sorter(i)
    delays, weights, N_syn = create_weight_and_delay('clustered', angle)
    param, E = restart_param()
    param['weights'] = weights
    param['N_syn'] = N_syn
    param['k_max'] = k_max
    E['K_c'] += k_max
    data_0 = simulation(param, E, delays, False, change = True)
    np.save(f'EK_effect/no_noise_clamp_data_{idx}_clu_{angle}_{k_max}', data_0)


if __name__ == '__main__':
    index = np.arange(0,100,1)
    pool = Pool(50)
    pool.map(run, index)
    pool.close()
    pool.join()
