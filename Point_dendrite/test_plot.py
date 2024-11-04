import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def tuning_vm(arr, s = 11, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    #  print(kappa, 'kappa')
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)

def create_weight_and_delay(regime, stim_alpha, i = 1):
    #  np.random.seed(i)
    N_syn  = np.array([7,8,9,10,11,12,13])
    N = int(np.random.choice(N_syn, 1))
    #  N = np.random.poisson(8,1)[0]
    #  if N < 2:
        #  N = 2
    weights = np.zeros(N)
    weights2 = np.zeros(N)
    delays = np.random.poisson(30, N) + 200
    #  stim_alpha = np.deg2rad(stim_alpha)
    for i in range(N):
        if regime == 'clustered':
            disp = 1/np.power(np.deg2rad(15)*2, 2)
            rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
            weights[i] = tuning_vm(np.rad2deg(rvs)/2, 11, stim_alpha)
        #  else:
        #      rvs = np.random.uniform(-np.pi,np.pi)
        #      #  rvs = np.random.uniform(-90,90)
        #      weights[i] = tuning_vm(rvs/2, 11, stim_alpha)
        #  rvs = np.random.uniform(-90,90)
        #      rvs = np.random.uniform(-180,180)
        #      weights[i] = tuning_vm(rvs, 11, stim_alpha)
        #  rvs = np.random.uniform(-180,180)
        #  weights2[i] = tuning_vm(rvs, 11, stim_alpha)
    #  print(np.mean(weights), stim_alpha)
    #  print(np.round(weights,2))
    #  return delays, weights*1, N, weights2
    return np.mean(weights)

bin1 = []
bin2 = []
bin3 = []
for i in range(1000):
    print(i)
    bin1.append(create_weight_and_delay('clustered', 0))
    bin2.append(create_weight_and_delay('clustered', 20))
    bin3.append(create_weight_and_delay('clustered', 45))

plt.hist(bin1, bins = 50, histtype = 'step')
plt.hist(bin2, bins = 50, histtype = 'step')
plt.hist(bin3, bins = 50, histtype = 'step')
plt.show()
#  create_weight_and_delay('clustered', 10)
#  create_weight_and_delay('clustered', 45)
