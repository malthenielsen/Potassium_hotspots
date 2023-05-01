import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')
from point_dendrite import *
from multiprocessing import Pool, cpu_count

def create_weight_and_delay_foldover(regime, stim_alpha, N_align, w):
    np.random.seed(299)
    weights = np.ones(N_align)*w
    delays = np.random.poisson(80, N_align) + 200
    #  for i in range(N_align):
    #      disp = 1/np.sqrt(np.radians(11/2))
    #      rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
    #      weights[i] = tuning_vm(rvs, 11, stim_alpha)
    return delays, weights, N_align

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

def run(N):
    #  mean = .7
    #  val = np.arange(mean - .2, mean + .2, .05)
    val = [.55, .6, .65]
    #  i = i/100 + .7
    #  fig, ax = plt.subplots(1,2, figsize = (10,7))
    for i in val:
        delays, weights, N_syn = create_weight_and_delay_foldover('clustered', 0, N, i)
        peak = []
        times = []
        k_max = np.linspace(4,10,100)
        NMDA = []
        for k in k_max:
            param, E = restart_param()
            param['weights'] = weights
            param['N_syn'] = N_syn
            E['K'] += k
            data = simulation(param, E, delays, False, change = False)
            V_baseline = np.mean(data['V'][10000:20000])
            peak.append(np.max(data['V']))
            Na = data['IG'][0,:]*2*np.pi*1e-4
            if np.max(Na) > 0.0002:
                NMDA.append(1)
            else:
                NMDA.append(0)
        np.save(f'limit_dir_illu/res_{np.round(i,2)}_{N}_301', peak)

def npick(i):
    if i < 10:
        return 7
    elif i < 20:
        return 8
    elif i < 30:
        return 9
    elif i < 40:
        return 10
    elif i < 50:
        return 11
    elif i < 60:
        return 12
    elif i < 70:
        return 13

def run(i):
    res = np.zeros(100)
    val = np.arange(0.1, 1.1, .1)
    N = npick(i)
    delays, weights, N_syn = create_weight_and_delay_foldover('clustered', 0, N, val[i%10])
    peak = []
    times = []
    k_max = np.linspace(6,18,100)
    NMDA = []
    for idx, k in enumerate(k_max):
        param, E = restart_param()
        param['weights'] = weights
        param['N_syn'] = N_syn
        E['K'] += k
        data = simulation(param, E, delays, False, change = False)
        V_baseline = np.mean(data['V'][10000:20000])
        res[idx] = np.max(data['V'])
        if res[idx] > -25:
            break
    np.save(f'./plane_plot/res_{np.round(val[i%10],2)}_{N}_301', res)


#  run(8)

if __name__ == '__main__':
    index = np.arange(0,70,1)
    pool = Pool(cpu_count() - 5)
    pool.map(run, index)
    pool.close()
    pool.join()
