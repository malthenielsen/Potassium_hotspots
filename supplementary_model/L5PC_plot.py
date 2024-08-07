import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from appstatpy.ExternalFunctions import *
plt.rcParams.update({'font.size': 14})
plt.rcParams['svg.fonttype'] = 'none'


def run(path):
    data = np.load(path)
    fps = []
    #  data = data[:,5000:45000]
    fig, ax = plt.subplots(1,3, figsize = (12,4))
    for i in range(36):
        #  plt.plot(data[0,:])
        #  plt.show()
        #  plt.plot(data[i,:])
        ax[i%3].plot(data[i,:])
        inds,_ = find_peaks(data[i,:], height = -25)
        #  ax[i%3].scatter(inds, data[i,inds])
        length = np.max(np.where(~np.isnan(data[i,:]))[0])
        #  print(len(inds), length)
        

        if length > 20000:
            fps.append(len(inds)/(length*1/40)*1000)
        else:
            fps.append(0)
    plt.show()
    fps = np.array(fps).reshape(-1,3)
    #  plt.show()
    return fps


x_val = np.linspace(0,1,22)
fig, ax = plt.subplots(1,1, figsize = (8,6))
#  colors = ['tab:green', 'tab:red', 'tab:blue']
colors = ['darkgrey', 'dodgerblue', 'goldenrod']
high = np.zeros((15,12))
medi = np.zeros((15,12))
lowi = np.zeros((15,12))

for i in range(15):
    #  path = f'dataL5PC2/dataL5PC/soma_{i}.npy'
    path = f'dataL5PC_weight/soma_{i}.npy'
    fps = run(path)
    high[i,:] = fps[:,0]
    medi[i,:] = fps[:,1]
    lowi[i,:] = fps[:,2]
    #  for j in range(3):
        #  ax.plot(x_val, fps[:,j], color = colors[j], alpha = .5)


def sigmoid(x, xoff, b, MAX):
    return MAX* 1/(1 + np.exp(b*(x - xoff)))


#  angles = np.linspace(0,35,9)
#  angles = np.append(angles, np.array([40, 50, 75, 90]))
angles = np.linspace(0,25,9)
angles = np.append(angles, np.array([30, 50, 90]))

#  ax.plot(angles, sigmoid(angles, 40, 0.2, 0.04))
#  plt.show()

con_angles = np.linspace(0,90,100)

p0, _ = curve_fit(sigmoid, angles, np.nanmean(high,axis=0), p0 = [20, 0.2, 10])
ax.plot(con_angles, sigmoid(con_angles, *p0), color = colors[0])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(medi,axis=0), p0 = [20, 0.2, 10])
ax.plot(con_angles, sigmoid(con_angles, *p0), color = colors[1])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(lowi,axis=0), p0 = [20, 0.2, 10])
ax.plot(con_angles, sigmoid(con_angles, *p0), color = colors[2])

ax.errorbar(angles, np.nanmean(lowi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[2], label = 'Low')
ax.scatter(angles, np.nanmean(lowi, axis = 0), color = colors[2])

ax.errorbar(angles, np.nanmean(medi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[1], label = 'Medium')
ax.scatter(angles, np.nanmean(medi, axis = 0), color = colors[1])

ax.errorbar(angles, np.nanmean(high, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[0], label = 'High')
ax.scatter(angles, np.nanmean(high, axis = 0), color = colors[0])

#  ax.legend(title ='$\Delta E_K$')

#  ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing freq [AU]', title = 'L5PC Neuron')
ax.set_xticks([0, 22.5, 45, 67.5, 90], [0, 22.5, 45, 67.5, 90])
ax.legend(['No shift', 'Low shift', 'High shift'], title = '$\Delta E_K$')
ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing frequency [Hz]', title = 'L5PC morphology')
plt.tight_layout()
#  fig.savefig('SuppF5', dpi = 200)
#  fig.savefig('SuppF5.svg', dpi = 200)

plt.show()





