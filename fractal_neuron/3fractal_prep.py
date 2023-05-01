import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
from scipy import stats
import sys
     
def tuning_vm(arr, s = .3, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    disp = 1/np.sqrt(np.deg2rad(s)/2)
    arr_r = np.linspace(-np.pi,np.pi, 1000)
    val = stats.vonmises.pdf(arr, disp, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, disp, loc = 0 + shift)
    #  plt.plot(arr_r, val_r)
    #  plt.show()
    return val / np.max(val_r)

def tuning_vm(arr, s = .3, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(np.deg2rad(arr), kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)

def random_segment(stim_alpha):
    n_syn = np.random.poisson(9)
    n_syn = np.random.choice([7,8,9,10,11,12,13])
    weights = np.zeros(n_syn)
    for i in range(n_syn):
        rvs = np.random.uniform(-np.pi,np.pi)
        weights[i] = tuning_vm(rvs, 11, stim_alpha)
        #  print(rvs, weights[i])
    return weights, n_syn

def clustered_segment(stim_alpha):
    n_syn = np.random.poisson(9)
    n_syn = np.random.choice([7,8,9,10,11,12,13])
    weights = np.zeros(n_syn)
    ori = np.zeros(n_syn)
    for i in range(n_syn):
        disp = 1/np.power(np.radians(11)*2)
        rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
        #weights[i] = tuning_vm(rvs, 11, stim_alpha)
        weights[i] = 1
        ori[i] = rvs
        #  print(rvs, weights[i])
    return weights, n_syn, ori

def mixed_segment(stim_alpha):
    n_syn = np.random.poisson(9)
    n_syn = np.random.choice([7,8,9,10,11,12,13])
    weights = np.zeros(n_syn)
    ori = np.zeros(n_syn)
    for i in range(n_syn):
        disp = 1/np.sqrt(np.radians(33))
        rvs = stats.vonmises.rvs(kappa = disp, loc = 0, size = 1)[0]
        #weights[i] = tuning_vm(rvs, 11, stim_alpha)
        weights[i] = 1
        ori[i] = rvs
        #  print(rvs, weights[i])
    return weights, n_syn, ori


def N_branches(N):
    return 2**(N+1)

def seg_lengths(N, min_L = 20):
    res = []
    res_t = []
    res_n = []
    for i in range(N):
        res.append(min_L*2**(N-i-1))
        res_t.append(min_L*2**(N-i-1)*N_branches(i))
        res_n.append(N_branches(i))

    return np.array(res), np.array(res_t), np.array(res_n)

res, rest, resn = seg_lengths(4)

stim_alpha = float(sys.argv[1])
print(stim_alpha, 'Stim alpha')
ID = int(sys.argv[3])

N_clusters = np.random.poisson(32, 1)
N_mixed = 0#np.random.poisson(20,1)

if sys.argv[2] == 'single':
    dend_clust = [30]
else:
    dend_clust = np.random.choice(np.arange(60,120,1),N_clusters, replace = False)
    mixed_clust = np.random.choice(np.arange(60,120,1),N_mixed, replace = False)

#  np.save('dendrite_cluster_loc', dend_clust)


def RF(d):
    left = [27,28,29,30,13,14,6]
    left_center = [23,24,25,26,11,12,5]
    right_center = [19,20,21,22,9,10,4]
    right = [15,16,17,18,7,8,3]
    center = [1,2]
    if d in left:
        time = 100
    if d in left_center:
        time = 400
    if d in right:
        time = 300
    if d in right_center:
        time = 200
    if d in center:
        time = 0
    return time + np.random.randint(-10,10,1)[0]
    #  return 0

'''
Here we create the clusters synapses
'''
char = np.zeros((1000,4))
stimV = []
iterator = 0
for d in dend_clust:
    weights, n_syn, ori = clustered_segment(stim_alpha)
    #  RF_split = RF(d)
    RF = np.random.randint(0,300,1)[0]
    for i in range(iterator, n_syn + iterator):
        char[i,0] = d
        char[i,1] = .99
        char[i,2] = weights[i-iterator]
        char[i,3] = ori[i-iterator]
        stim = np.random.poisson(30,1)
        stim_vec = np.hstack([stim[0] + 500, stim[0] + 800, stim[0] + 1200])
        stim_vec += RF
        stimV.append(stim_vec)
    iterator += n_syn

for d in mixed_clust:
    weights, n_syn, ori = mixed_segment(stim_alpha)
    #  RF_split = RF(d)
    RF = np.random.randint(0,300,1)[0]
    for i in range(iterator, n_syn + iterator):
        char[i,0] = d
        char[i,1] = .99
        char[i,2] = 0*weights[i-iterator]
        char[i,3] = ori[i-iterator]
        stim = np.random.poisson(30,1)
        stim_vec = np.hstack([stim[0] + 500, stim[0] + 800, stim[0] + 1200])
        stim_vec += RF
        stimV.append(stim_vec)
    iterator += n_syn



'''
Below here we make random synapses, their weight is set at 0, so they wont interfeere.
Kept it in the script if ever needed
'''

p = np.array([1/2, 1/2,
              1/4, 1/4, 1/4, 1/4,
              1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8,
              1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16,
              1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16])
p /= np.sum(p)

for i in range(iterator, 1000):
    d = np.random.choice(np.arange(0,30,1), p = p)
    loc = np.random.uniform(1e-3, 1, 1)
    ori = (np.random.uniform(-np.pi,np.pi,1))
    weight = 0#tuning_vm(ori,11, stim_alpha) * tuning_vm(ori,11, 0)
    char[i,0] = d
    char[i,1] = loc
    char[i,2] = weight
    char[i,3] = ori = 100
    N_fir = np.random.poisson(3,1)
    if N_fir == 0:
        N_fir += 1
    stim = np.random.randint(0,2000, N_fir)
    stimV.append(stim)

VecStim = np.zeros((1000,15))

for i in range(len(stimV)):
    VecStim[i,:len(stimV[i])] = stimV[i]

np.save(f'bin/3VecFrac_{ID}', VecStim)
np.save(f'bin/3CharFrac_{ID}', char)

#  fig, ax = plt.subplots(1,1, figsize = (8,6))
#  ax.hist(

#  for i in range(3000):
#      print(char[i,:])


    



