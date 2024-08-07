import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('K_PAPER')
from point_dendrite import *
from multiprocessing import Pool, cpu_count
import tqdm

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

#  delays, weights, N_syn = create_weight_and_delay_foldover('clustered', 0, 10, .3)
#  param, E = restart_param()
#  param['weights'] = weights
#  param['N_syn'] = N_syn
#  E['K'] += 1
#  data = simulation(param, E, delays, True, change = False)
#
#  exit()

def run(i):
    N = 8
    mean = .7
    val = np.arange(mean - .2, mean + .2, .05)
    #  print(len(val))
    #  exit()
    #  val = [.55, .6, .65]
    #  i = i/100 + .7
    #  fig, ax = plt.subplots(1,2, figsize = (10,7))
    #  for i in val:
        #  print(i)
    delays, weights, N_syn = create_weight_and_delay_foldover('clustered', 0, N, val[i])
    #  _, weights2, _ = create_weight_and_delay_foldover('clustered', 0, N, i)
    peak = []
    times = []
    k_max = np.linspace(4,10,100)
    NMDA = []
    for k in (k_max):
        param, E = restart_param()
        param['weights'] = weights
        param['weights2'] = weights*0
        param['N_syn'] = N_syn
        E['K'] += k
        data = simulation(param, E, delays, False, change = False)
        V_baseline = np.mean(data['V'][10000:20000])
        #  ax[0].plot(data['RT'], data['V'] - V_baseline, color = plt.cm.cool(k/10))
        #  ax[0].set_ylim(-10,50)
        #  ax[1].scatter(k, np.max(data['V'] - V_baseline), color = 'black')
        peak.append(np.max(data['V']) - V_baseline)
        Na = data['IG'][0,:]*2*np.pi*1e-4
        if np.max(Na) > 0.0002:
            NMDA.append(1)
        else:
            NMDA.append(0)
        #  threshold = -20
        #  if np.max(data['V']) > threshold: # Traces bellow thr is just ignored, as it it not considered a spike
        #      first = (np.where(data['V'] > threshold)[0][0])
        #      last = np.where(data['V'][first + 100:] < threshold)[0][0]
        #      t_dur = data['RT'][last + first + 100] - data['RT'][first]
        #      times.append(t_dur)
        #  else:
        #      times.append(0)
    #  plt.show()
    np.save(f'limit_dir_illu/res_{np.round(i,2)}_{N}_2024', peak)
        #  np.save(f'limit_dir_monday/nmda_{np.round(i,2)}_{N}_301', NMDA)
        #  np.save(f'limit_dir_night/time_{np.round(i,2)}_{N}_301', times)

#  run(8)
#  exit()

if __name__ == '__main__':
    index = np.arange(0,8,1)
    pool = Pool(8)
    pool.map(run, index)
    pool.close()
    pool.join()

exit()
#
def plot(N, val):
    print(val)
    #  val = .6
    k_max = np.array([0,6,12,18])
    arr = np.zeros((4,50000))
    #  for i in range(5):
        #  np.random.seed(i)
    delays, weights, N_syn = create_weight_and_delay_foldover('clustered', 0, N, val)
    idx = 0
    for k in k_max:
        param, E = restart_param()
        param['weights'] = weights
        param['N_syn'] = N_syn
        E['K'] += k
        data = simulation(param, E, delays, False, change = False)
        V_baseline = data['V'][:50000]# - np.mean(data['V'][10000:20000])
        arr[idx,:] = V_baseline
        idx += 1
        #  plt.plot(data['V'] - V_baseline)
    np.save(f'transition_to_nonlinear_{val}', arr)


#  plot(8, .65)
#  plot(8, .7)
#  plot(8, .55)
#  plot(8, .5)
#  plot(8, .6)
#  plot(8, .8)

#  plot(8, .65)
#  plot(8, .7)
#  plot(8, .55)
#  plot(8, .5)
#  plot(8, .6)
#  plot(8, .8)
#  plot(8, .45)
#  plot(8, .4)

data_6 = np.load('transition_to_nonlinear_0.6.npy')
data_65 = np.load('transition_to_nonlinear_0.65.npy')
data_55 = np.load('transition_to_nonlinear_0.55.npy')
data_5 = np.load('transition_to_nonlinear_0.5.npy')
data_7 = np.load('transition_to_nonlinear_0.7.npy')
data_8 = np.load('transition_to_nonlinear_0.8.npy')
data_4 = np.load('transition_to_nonlinear_0.4.npy')
data_45 = np.load('transition_to_nonlinear_0.45.npy')

dat_arr = [data_4, data_45, data_5, data_55, data_6, data_65, data_7, data_8]
fig, ax = plt.subplots(1,3, figsize = (12,6), sharey = True, sharex = False)
cobalt = '#0047AB'
terra  = '#E3735E' 
tiffany = '#0ABAB5'
Norm = np.linspace(0.2,1,9, endpoint = True)
Norm = np.delete(Norm, -2)
#  Norm = np.delete(Norm, 0)
print(Norm)
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=.2, vmax=.8)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno_r)
cmap.set_array([])
for axi in ax.ravel():
    axi.spines['top'].set_visible(False)
    axi.spines['bottom'].set_visible(False)
    axi.spines['left'].set_visible(False)
    axi.spines['right'].set_visible(False)
    if axi != ax[2]:
        #  axi.set_yticks([])
        axi.set_xticks([])
ax[2].spines['left'].set_visible(True)
ax[2].spines['bottom'].set_visible(True)
for i in range(6):
    ax[0].plot(dat_arr[i][0,25000:42000]- np.mean(dat_arr[i][0,10000:15000]), color = plt.cm.inferno_r(Norm[i]))
    #  ax[1].plot(dat_arr[i][1,25000:42000]- np.mean(dat_arr[i][1,10000:15000]), color = plt.cm.inferno_r(Norm[i]))
    ax[1].plot(dat_arr[i][1,25000:42000]- np.mean(dat_arr[i][1,10000:15000]), color = plt.cm.inferno_r(Norm[i]))
    ax[2].plot(dat_arr[i][3,25000:42000]- np.mean(dat_arr[i][3,10000:15000]), color = plt.cm.inferno_r(Norm[i]))

#  ax[0].legend([.55, .6, .65], title = 'Mean spine input')

ax[0].text(-0.1, 1.05, 'E', fontweight = 'bold', transform = ax[0].transAxes, fontsize = 20)
ax[0].set_title('$\Delta E_K = 0mV$')
#  ax[1].set_title('$\Delta E_K = 6mV$')
ax[1].set_title('$\Delta E_K = 6mV$')

ax[2].set_title('$\Delta E_K = 18mV$')
fig.colorbar(cmap, label = 'Mean spine input')
plt.savefig('transition_fig', dpi = 200)
    
plt.savefig('FIG_2E.svg', dpi = 400)
plt.savefig('FIG_2E.pdf', dpi = 400)

plt.show()




#  data = np.load('transition_to_nonlinear.npy')
#  k_max = np.array([0, 4,6,8,10])
#  fig, ax = plt.subplots(1,1, figsize = (6,6))
#  colors = ['hotpink','goldenrod', 'olive', 'steelblue', 'teal']
#  for i in range(5):
#      ax.plot(data[i,25000:42000], color = colors[i], lw = 1.5)
#  ax.set(xticks = [], yticks = [], ylabel = 'Vm', xlabel = 'Time')
#  ax.legend(k_max, title = '$\Delta E_K [mV]$')
#  plt.savefig('transition_fig')
#
#  plt.show()
#
#
#
