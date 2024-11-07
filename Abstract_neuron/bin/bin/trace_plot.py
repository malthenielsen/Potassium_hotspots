import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from ExternalFunctions import Chi2Regression
from iminuit import Minuit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

def rm(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')


def func(path):
    #  fnames = fnames[:5]

    #  fnames_soma = fnames[1::2]
    fnames_trunk = glob(f'{path}/trunk?*.npy')
    fnames_trunk.sort(key=os.path.getctime)


    colors = ['darkgrey', 'dodgerblue', 'goldenrod']

    data = np.zeros((3,8,len(fnames_trunk)))


    for l, fname in enumerate(fnames_trunk):
        trunk = np.load(fname)[::-1]
        bin_ = []
        
        if l == 0:
            fig, ax = plt.subplots(3,1, figsize = (7,6), sharey = True, sharex = True)
            for i in range(len(trunk)):
                inds, _ = find_peaks(trunk[i], height = -30)
                #  for idx in inds:
                #      trunk[i][idx - 55:idx + 55] = np.mean(trunk[i][idx+55:idx+60])

                if i%3 == 0:
                    baseline = np.mean(trunk[i][1000:2000])
                    print(trunk[i].shape)
                    #  fig, ax = plt.subplots(1,1, figsize = (7,6))
                trunk_tmp = trunk[i][15000:70000] - baseline
                #  trunk_tmp[trunk_tmp > 30] = 30
                area = np.trapz(trunk_tmp, dx = .001)
                bin_.append(area)
                #  ax.plot(rm(trunk[i] - baseline, 3000), color = colors[i], lw = 2)
                #  ax.plot(rm(trunk[i][15000:70000], 1), color = colors[i], lw = 2)
                #  ax[i%3].plot(rm(trunk[i][:], 1), color = colors[i%3], lw = 2)
                import matplotlib as mpl
                time = np.linspace(0,100,4000)
                ax[i%3].plot(time, rm(trunk[i][14000:18000], 1), color = plt.cm.cool_r((i//3)/8), lw = 2)
                cbaxes = inset_axes(ax[0], width="5%", height="200%", loc = 5)
                plt.colorbar(plt.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin=600, vmax=1200), cmap = 'cool_r'), orientation = 'vertical',cax=cbaxes, label = '$Length$')
    plt.show()
                ax.text(-0.1, 1.05, 'E', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
                if i%3 == 2:
                    ax.legend(['No shift', 'Low', 'High'], title = '$\Delta E_k$')
                    ax.set(xlabel = 'Time[AU]', ylabel = 'Trunk Vm minus baseline')
                    #  ax.set_xticks([])
                    #  ax.set_yticks([])
                    #  plt.savefig('AOC_plot', dpi = 200)
                plt.savefig('FIG_4E_CB.svg', dpi = 400)
                plt.savefig('FIG_4E_CB.pdf', dpi = 400)
                plt.show()


                exit()

        data[:,:,l] = np.array(bin_).reshape(8,3).T

    t_test = np.zeros((2,8))
    residuals_arr = np.zeros((2,8,len(fnames_soma)))
    T_large = np.zeros(2)
    R_val = np.zeros(2)


    for i in range(8):
        for j in range(2):
            residuals = np.diff(data[[0,j+1], i, :], axis = 0).ravel()
            residuals_arr[j,i,:] = residuals
            sdom = np.std(residuals, ddof = 1)/np.sqrt(len(residuals))
            t = np.mean(residuals) / sdom
            #  print(stats.t.sf(t, len(residuals)-1))
            t_test[j,i] = stats.t.sf(t, len(residuals)-1)
    for i in range(2):
        residuals = residuals_arr[i,:,:].ravel()
        sdom = np.std(residuals, ddof = 1)/np.sqrt(len(residuals))
        t = np.mean(residuals) / sdom
        T_large[i] = stats.t.sf(t, len(residuals)-1)
        R_val[0] = t
        R_val[1] = len(residuals)-1
    mean = np.mean(data, axis = 2)
    eom = np.std(data, axis = 2)/np.sqrt(data.shape[2])
    return mean, eom, t_test, residuals_arr, T_large, R_val

paths = ['./new_tt/*.npy']#,
#  paths = ['./folder_trunk_test/']#,
        # './no_cluster/*.npy']
trunk_lenghts = np.arange(100,900,100) + 300

#  func(paths[0])
mean, eom, t_test, residuals, T_large, R_val = func(paths[0])
#  mean_n, eom_n, t_test, residuals, T_large, R_val = func(paths[1])

fig, ax = plt.subplots(1,1, figsize = (7,6))
ax.text(-0.1, 1.05, 'G', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
def func(x,a,b):
    return a*x + b

colors = ['darkgrey', 'dodgerblue', 'goldenrod']

for i in range(3):
    fit_chi2 = Chi2Regression(func, trunk_lenghts, mean[i,:], eom[i:])
    minuit_chi2 = Minuit(fit_chi2, a=0.0, b=0.0)
    minuit_chi2.errordef = 1.0
    minuit_chi2.migrad()
    ax.plot(trunk_lenghts, func(trunk_lenghts, *minuit_chi2.values[:]), color = colors[i])
    print(minuit_chi2.values[:])
    print(minuit_chi2.errors[:])
    print(stats.chi2.sf(minuit_chi2.fval, len(trunk_lenghts) - 2))

#  for i in range(3):
    #  fit_chi2 = Chi2Regression(func, trunk_lenghts, mean_n[i,:], eom_n[i:])
    #  minuit_chi2 = Minuit(fit_chi2, a=0.0, b=0.0)
    #  minuit_chi2.errordef = 1.0
    #  minuit_chi2.migrad()
    #  ax.plot(trunk_lenghts, func(trunk_lenghts, *minuit_chi2.values[:]), color = colors[i], ls = '--')
    #  print(minuit_chi2.values[:])
    #  print(minuit_chi2.errors[:])



ax.errorbar(trunk_lenghts, mean[0,:], eom[0,:], ls = '', capsize = 4, color = colors[0], marker = 'D')
ax.errorbar(trunk_lenghts, mean[1,:], eom[1,:], ls = '', capsize = 4, color = colors[1], marker = 'D')
ax.errorbar(trunk_lenghts, mean[2,:], eom[2,:], ls = '', capsize = 4, color = colors[2], marker = 'D')

#  ax.errorbar(trunk_lenghts, mean_n[0,:], eom_n[0,:], ls = '', capsize = 4, color = colors[0], marker = 'x')
#  ax.errorbar(trunk_lenghts, mean_n[1,:], eom_n[1,:], ls = '', capsize = 4, color = colors[1], marker = 'x')
#  ax.errorbar(trunk_lenghts, mean_n[2,:], eom_n[2,:], ls = '', capsize = 4, color = colors[2], marker = 'x')
ax.legend(['No shift', 'Low shift', 'High shift'], title = '$\Delta E_K$')
ax.set(xlabel = 'Trunk lenghts[um]', ylabel = 'AOC [AU]', title = 'AOC as function of trunklenght')
#  plt.savefig('AOC_compar', dpi = 200)
plt.savefig('FIG_4G.svg', dpi = 400)
plt.savefig('FIG_4G.pdf', dpi = 400)
plt.show()








