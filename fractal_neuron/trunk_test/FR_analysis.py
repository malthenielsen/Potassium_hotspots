import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from ExternalFunctions import Chi2Regression
from iminuit import Minuit

def rm(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

trunk_lengths = np.arange(400, 1500, 100) + 300
trunk_lengths = np.append(trunk_lengths, np.array([2000, 2500, 3000, 4000]))
N_t = len(trunk_lengths)

def func(path):
    #  fnames = glob(path)
    #  fnames.sort(key=os.path.getctime)
    #  fnames = fnames[:5]

    #  fnames_soma = fnames[1::2]
    #  fnames_trunk = fnames
    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_soma.sort(key=os.path.getctime)
    print(fnames_soma)


    colors = ['darkgrey', 'dodgerblue', 'goldenrod']

    data = np.zeros((3,N_t,len(fnames_soma)))


    for l, fname in enumerate(fnames_soma):
        soma = np.load(fname)
        bin_ = []
        
        for i in range(len(soma)):
            inds, _ = find_peaks(soma[i], height = 40)
            #  print(len(inds))
            if np.mean(soma[i]) > 20:
                bin_.append(np.nan)
            else:
                bin_.append(len(inds))
            #  plt.plot(soma[i])
            #  if i%3 ==2:
                #  plt.title(trunk_lengths[i//3])
                #  plt.show()
        fin = np.array(bin_).reshape(N_t,3).T
        for idx in range(N_t):
            if any(np.isnan(fin[:,idx])):
                fin[:,idx] = np.nan
        data[:,:,l] = fin
    t_test = np.zeros((2,N_t))
    residuals_arr = np.zeros((2,N_t,len(fnames_soma)))
    T_large = np.zeros(2)
    R_val = np.zeros(2)


    for i in range(N_t):
        for j in range(2):
            residuals = np.diff(data[[0,j+1], i, :], axis = 0).ravel()
            residuals_arr[j,i,:] = residuals
            sdom = np.std(residuals, ddof = 1)/np.sqrt(len(residuals))
            t = np.mean(residuals) / sdom
            #  print(stats.t.sf(t, len(residuals)-1))
            t_test[j,i] = stats.t.sf(t, len(residuals)-1)

    for i in range(2):
        residuals = residuals_arr[i,:,:].ravel()
        sdom = np.nanstd(residuals, ddof = 1)/np.sqrt(len(residuals))
        t = np.nanmean(residuals) / sdom
        T_large[i] = stats.t.sf(t, len(residuals)-1)
        R_val[0] = t
        R_val[1] = len(residuals)-1

    print(T_large)
    exit()


    mean = np.nanmean(data, axis = 2)
    print(mean)
    eom = np.nanstd(data,axis =2)/np.sqrt(data.shape[-1])
    return mean, eom 

#  paths = ['./new_tt/*.npy']#,
paths = ['./new_folder_trunk_test/']#,
        # './no_cluster/*.npy']

mean, eom = func(paths[0])
#  exit()
#  mean_n, eom_n, t_test, residuals, T_large, R_val = func(paths[1])

fig, ax = plt.subplots(1,1, figsize = (7,6))
ax.text(-0.1, 1.05, 'G', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)

def func(x,a,b):
    return a*x + b

def func(x,a,b):
    return np.exp((x-b)*a)

#  def func(x,a,b,c):
    #  return c*(x-b)**a

colors = ['darkgrey', 'dodgerblue', 'goldenrod']

for i in range(3):
    mask = eom[i,:] > 0
    fit_chi2 = Chi2Regression(func, trunk_lengths[mask], mean[i,mask], eom[i,mask])
    minuit_chi2 = Minuit(fit_chi2, a=-.001, b=1450.)
    minuit_chi2.errordef = 1.0
    #  minuit_chi2.fixed['b'] = True
    minuit_chi2.migrad()
    ax.plot(np.linspace(700,4000, 1000), func(np.linspace(700,4000, 1000), *minuit_chi2.values[:]), color = colors[i])
    #  ax.plot(trunk_lengths, func(trunk_lengths, -0.15, 699, 25), color = colors[i])
    print(minuit_chi2.values[:])
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(trunk_lengths) - 2))
    print(np.log(2)/minuit_chi2.values['a'])

#  for i in range(3):
    #  fit_chi2 = Chi2Regression(func, trunk_lengths, mean_n[i,:], eom_n[i:])
    #  minuit_chi2 = Minuit(fit_chi2, a=0.0, b=0.0)
    #  minuit_chi2.errordef = 1.0
    #  minuit_chi2.migrad()
    #  ax.plot(trunk_lengths, func(trunk_lengths, *minuit_chi2.values[:]), color = colors[i], ls = '--')
    #  print(minuit_chi2.values[:])
    #  print(minuit_chi2.errors[:])



ax.errorbar(trunk_lengths, mean[0,:], eom[0,:], ls = '', capsize = 0, color = colors[0], marker = '')
ax.errorbar(trunk_lengths, mean[1,:], eom[1,:], ls = '', capsize = 0, color = colors[1], marker = '')
ax.errorbar(trunk_lengths, mean[2,:], eom[2,:], ls = '', capsize = 0, color = colors[2], marker = '')
ax.set_xticks([700, 1200, 1700, 2200, 2700, 3200, 3700],[700, 1200, 1700, 2200, 2700, 3200, 3700])

#  ax.errorbar(trunk_lenghts, mean_n[0,:], eom_n[0,:], ls = '', capsize = 4, color = colors[0], marker = 'x')
#  ax.errorbar(trunk_lenghts, mean_n[1,:], eom_n[1,:], ls = '', capsize = 4, color = colors[1], marker = 'x')
#  ax.errorbar(trunk_lenghts, mean_n[2,:], eom_n[2,:], ls = '', capsize = 4, color = colors[2], marker = 'x')
ax.legend(['No shift', 'Low shift', 'High shift'], title = '$\Delta E_K$')
ax.set(xlabel = 'Trunk lenghts[um]', ylabel = 'AOC [AU]', title = 'AOC as function of trunklenght')
#  plt.savefig('AOC_compar', dpi = 200)
plt.savefig('FIG_4H.svg', dpi = 400)
plt.savefig('FIG_4H.pdf', dpi = 400)
plt.show()

#  np.save('FR_mean', mean)
#  np.save('FR_eom', eom)








