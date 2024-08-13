import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
from iminuit import Minuit
import matplotlib.gridspec as gridspec


from appstatpy.ExternalFunctions import *
plt.rcParams.update({'font.size': 14})
plt.rcParams['svg.fonttype'] = 'none'


def run(path):
    data = np.load(path)
    print(data.shape)
    fps = []
    #  data = data[:,5000:45000]
    #  fig, ax = plt.subplots(1,3, figsize = (12,4))
    for i in range(66):
        #  plt.plot(data[0,:])
        #  plt.show()
        #  plt.plot(data[i,:])
        #  ax[i%3].plot(data[i,:])
        inds,_ = find_peaks(data[i,:], height = 0)
        #  inds,_ = find_peaks(data[i,:], height = -10)
        #  ax[i%3].scatter(inds, data[i,inds])
        length = np.max(np.where(~np.isnan(data[i,:]))[0])
        #  print(len(inds), length)
        

        #  if length > 10000:
        fps.append(len(inds)/(length*1/40)*1000)
        #  else:
            #  fps.append(0)
    #  plt.show()
    fps = np.array(fps).reshape(-1,3)
    #  plt.show()
    return fps


x_val = np.linspace(0,1,22)
#  fig, ax = plt.subplots(1,1, figsize = (8,6))

fig = plt.figure(figsize=(7, 5))  # Adjust the figure size as needed

# Create a GridSpec with 1 row and 3 columns (2+1=3 to get the width ratio of 2:1)
gs = gridspec.GridSpec(1, 3)

# Create the first subplot (twice the width)
ax1 = fig.add_subplot(gs[0, 0:2])  # This occupies the first two columns
ax1.set_title('Window 1 (2x width)')

# Create the second subplot (normal width)
ax2 = fig.add_subplot(gs[0, 2])  # This occupies the third column
ax2.set_title('Window 2 (1x width)')
#  colors = ['tab:green', 'tab:red', 'tab:blue']
colors = ['darkgrey', 'dodgerblue', 'goldenrod']
high = np.zeros((15,22))
medi = np.zeros((15,22))
lowi = np.zeros((15,22))

for i in range(15):
    #  path = f'dataL5PC/dataL5PC/soma_{i}.npy'
    path = f'data_weight/soma_{i}.npy'
    #  path = f'data_weight_high/data_weight/soma_{i}.npy'
    #  path = f'data/data/soma_{i}.npy'
    #  path = f'dataSUPP/AUG/soma_{i}.npy'
    #  path = f'data_weight/data_weight/soma_{i}.npy'
    #  path = f'gain/hight_weight/soma_{i}.npy'
    #  path = f'data/SKv3/soma_{i}.npy'
    #  path = f'data/SKE2/soma_{i}.npy'
    fps = run(path)
    high[i,:] = fps[:,0]
    medi[i,:] = fps[:,1]
    lowi[i,:] = fps[:,2]
    #  for j in range(3):
        #  ax.plot(x_val, fps[:,j], color = colors[j], alpha = .5)


def sigmoid(x, xoff, b, MAX):
    return MAX* 1/(1 + np.exp(b*(x - xoff)))


#  angles = np.linspace(0,25,20)
angles = np.linspace(0,35,18)
angles = np.append(angles, np.array([40, 50, 75, 90]))
con_angles = np.linspace(0,90,100)

p0, _ = curve_fit(sigmoid, angles, np.nanmean(high,axis=0), p0 = [20, 0.2, 10])
ax1.plot(con_angles, sigmoid(con_angles, *p0), color = colors[0])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(medi,axis=0), p0 = [20, 0.2, 10])
ax1.plot(con_angles, sigmoid(con_angles, *p0), color = colors[1])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(lowi,axis=0), p0 = [20, 0.2, 10])
ax1.plot(con_angles, sigmoid(con_angles, *p0), color = colors[2])

ax1.errorbar(angles, np.nanmean(lowi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[2], label = 'High ')
ax1.scatter(angles, np.nanmean(lowi, axis = 0), color = colors[2])

ax1.errorbar(angles, np.nanmean(medi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[1], label = 'Low ')
ax1.scatter(angles, np.nanmean(medi, axis = 0), color = colors[1])

ax1.errorbar(angles, np.nanmean(high, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[0], label = 'None ')
ax1.scatter(angles, np.nanmean(high, axis = 0), color = colors[0])

#  ax.legend(title ='$\Delta E_K$')

#  ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing freq [AU]', title = 'L5PC Neuron')
#  ax1.set_xticks([0, 22.5, 45, 67.5, 90], [0, 22.5, 45, 67.5, 90])
ax1.legend(title = r'$\Delta E_{K^+}$ shift')
ax1.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing frequency [Hz]', title = 'Abstract neuron morphology')

p0, _ = curve_fit(sigmoid, angles, np.nanmean(high,axis=0), p0 = [20, 0.2, 10])
ax2.plot(con_angles, sigmoid(con_angles, *p0), color = colors[0])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(medi,axis=0), p0 = [20, 0.2, 10])
ax2.plot(con_angles, sigmoid(con_angles, *p0), color = colors[1])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(lowi,axis=0), p0 = [20, 0.2, 10])
ax2.plot(con_angles, sigmoid(con_angles, *p0), color = colors[2])

ax2.errorbar(angles, np.nanmean(lowi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[2], label = 'Increase ')
ax2.scatter(angles, np.nanmean(lowi, axis = 0), color = colors[2])

ax2.errorbar(angles, np.nanmean(medi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[1], label = 'Initial ')
ax2.scatter(angles, np.nanmean(medi, axis = 0), color = colors[1])

ax2.errorbar(angles, np.nanmean(high, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[0], label = 'Decrease ')
ax2.scatter(angles, np.nanmean(high, axis = 0), color = colors[0])

#  ax.legend(title ='$\Delta E_K$')

#  ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing freq [AU]', title = 'L5PC Neuron')
#  ax2.set_xticks([0, 22.5, 45, 67.5, 90], [0, 22.5, 45, 67.5, 90])
ax2.legend(title = r'Synaptic weight shift')
ax2.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing frequency [Hz]', title = 'L5PC neuron morphology')

ax1.set_xlim(-0.5,30)
ax2.set_xlim(80,90)
plt.tight_layout()
fig.savefig('Supp_weight_test.pdf', dpi = 200)
#  fig.savefig('SuppF5.svg', dpi = 200)
#  fig.savefig('4c.pdf', dpi = 300)
#  plt.close(fig)

plt.show()

def fn_add(x, a):
    return x + a

def fn_mul(x,a):
    return x*a

#  fig, ax = plt.subplots(1,2, figsize = (12,5))

low = np.nanmean(high, axis =0)
high = np.nanmean(lowi, axis =0)
medi = np.nanmean(medi, axis =0)
lowi = low

#  p0_high_add, _ = curve_fit(fn_add, lowi, high , p0 = [1])
#  p0_high_mul, _ = curve_fit(fn_mul, lowi, high , p0 = [1])
#  p0_medi_add, _ = curve_fit(fn_add, lowi, medi , p0 = [1])
#  p0_medi_mul, _ = curve_fit(fn_mul, lowi, medi , p0 = [1])
#  print(p0_high_add)
#  print(p0_high_mul)
#  print(p0_medi_add)
#  print(p0_medi_mul)
#
#  print(stats.ttest_rel( high,lowi, nan_policy = 'omit', alternative = 'greater'))
#
#  print(stats.ttest_rel( medi,lowi, nan_policy = 'omit', alternative = 'greater'))
#
#  ax[0].errorbar(angles, lowi, yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[0])
#  ax[0].errorbar(angles, high, yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[2])
#  ax[0].plot(angles, lowi*p0_high_mul[0], color = 'red', label = 'Multiplicative')
#  ax[0].plot(angles, lowi + p0_high_add[0], color = 'green', label = 'Addiative')
#  ax[0].legend()
#
#  #  ax[1].errorbar(lowi, medi, yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), xerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '')
#
#  ax[1].errorbar(lowi, high, yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), xerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '')
#
#  ax[1].plot(lowi, fn_add(lowi, *p0_high_add), color = 'green')
#  ax[1].plot(lowi, fn_mul(lowi, *p0_high_mul), color = 'red')
#
#
#
#  plt.show()


def OSI(arr, ori):
    if np.mean(arr) == 0:
        return np.nan
    ori = np.deg2rad(ori)
    #  print(ori, arr)
    top = np.sum(arr * np.exp(2*1j*ori))
    F = top/np.sum(arr)
    return 1  - np.arctan2(np.imag(F), np.real(F))

#  #  angles = np.linspace(0,35,18)
#  #  angles = np.append(angles, np.array([40,50,75,90]))
#  fig, ax = plt.subplots(figsize = (5,6))
#  ax2 = ax.twinx()
#  #  color = ['teal', 'goldenrod', 'teal', '#75151E','#0047AB' ]
#  ls = ['solid', '--', '-.', ':', (0,(1,10))]
#  labels = ['50/50', '75/25', '25/75', '0/100', '100/0']
#  paths = [lowi, medi, high]
#  OSI_arr = np.zeros(3)
#  max_arr = np.zeros(3)
#
#  for idx , path in enumerate(paths):
#      #  means, data = func(path)
#      #  for i in range(3):
#      OSI_arr[idx] = OSI(path, angles)
#      max_arr[idx] = path[0]
#  x_arr = np.arange(3)
#  ax2.scatter(x_arr, OSI_arr, color = 'slategrey')
#  ax2.plot(x_arr, OSI_arr, color = 'slategrey', label = labels[idx])
#  ax.scatter(x_arr, max_arr, color = 'black')
#  ax.plot(x_arr, max_arr, color = 'black')
#      #  ax2.scatter(x_arr, fwhm, ls = ls[idx], color = 'red')
#      #  ax2.plot(x_arr,    fwhm, ls = ls[idx], color = 'red')
#      #  print(means[:,0])
#      #  print(idx)
#  ax2.set_ylim(0,1)
#  ax.set_ylim(0,45)
#  #  ax2.set_ylim(0,1)
#  #  ax.set_ylim(11,13.5)
#  ax.set_ylabel('Peak firing rate (spikes/s)', color = 'black')
#  ax2.set_ylabel('Orientation sensitivity Index', color = 'slategrey')
#  ax2.set_xticks([0,1,2],['None', r'Low', r'High'] , rotation = 30)
#  plt.tight_layout()
#  fig.savefig('OSI', dpi =  400)
#
#
#  plt.show()

def fn(FRA, FRB, eFRB, eFRA):
    mask = FRA > 0
    fig1, ax = plt.subplots(1,2, figsize  = (14,6))
    mul_err = np.sqrt(((1/FRA)**2*eFRB**2 + (FRB/FRA**2)**2*eFRA**2))
    add_err = np.sqrt(eFRB**2 + eFRA**2)
    add_val = FRB - FRA
    mul_val = FRB/FRA
    ax1 = ax[0]
    ax3 = ax[1]


    def add(x,a):
        return x + a

    def mul(x,a):
        return x * a

    def fit(x, a):
        return a

    #  ax1.errorbar(angles, FRA, yerr = eFRA, ls = '', color = 'green')
    ax1.errorbar(angles, FRB, yerr = eFRB, ls = '', color = 'black', label  = 'Low $\Delta E_K$' )
    ax1.scatter(angles, FRB,  color = 'black')

    #  ax1.set_ylim(0,3)
    #  ax2 = ax1.twinx()
    #  ax2.errorbar(angles, FRB - FRA, yerr = add_err, ls = '', marker = 'o', color = 'black')

    chi2_fit = Chi2Regression(fit, add_val[mask], add_val[mask], add_err[mask])
    minuit_chi2 = Minuit(chi2_fit, a = 1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    print('add')
    print(minuit_chi2.values)
    print(minuit_chi2.errors)
    pval = stats.chi2.sf(minuit_chi2.fval, len(angles[mask]))
    print(pval)
    ax[0].text(10,10, r'$\xi_{Add}\ = \ $' + str(np.round(minuit_chi2.values[0],2)) + '$\pm$' + str(np.round(minuit_chi2.errors[0],2)))

    ax1.plot(angles, FRA + minuit_chi2.values['a'], color = 'green', label = 'Addiative') 
    ax3.plot(FRA, FRA + minuit_chi2.values['a'], color = 'green', label = 'Addiative')

    chi2_fit = Chi2Regression(fit, mul_val[mask], mul_val[mask], mul_err[mask])
    minuit_chi2 = Minuit(chi2_fit, a = 1)
    minuit_chi2.errordef = 1
    minuit_chi2.migrad()
    print('mul')
    print(minuit_chi2.values)
    print(minuit_chi2.errors)
    pval = stats.chi2.sf(minuit_chi2.fval, len(angles[mask]))
    ax[0].text(10,9, r'$\xi_{Mul}\ = \ $' + str(np.round(minuit_chi2.values[0],2)) + '$ \pm$ ' + str(np.round(minuit_chi2.errors[0],2)))
    print(pval)

    ax1.plot(angles, FRA*minuit_chi2.values['a'], color = 'blue', label = 'Multiplicative') 
    ax3.plot(FRA, FRA * minuit_chi2.values['a'], color = 'blue', label = 'Multiplicative')


    ax3.errorbar(FRA, FRB, yerr = eFRA, xerr =eFRB, ls = '', label = 'Data', color = 'slategrey')
    ax1.set(xlabel = '$\Delta$ Target orientation ($\circ$)', ylabel = 'Firing range (spikes/s)', title = 'Soma firing range tuning') 
    ax3.set(xlabel = 'Firing rate before (spikes/s)', ylabel = 'Firing rate after (spikes/s)', title = 'Fit to data')
    ax1.legend()
    ax3.legend()
    #  fig1.savefig('S6new.png', dpi = 300)
    #  plt.show()

fn(low, medi, np.nanstd(fps, axis = 1)/np.sqrt(10), np.nanstd(fps, axis = 1)/np.sqrt(10))
