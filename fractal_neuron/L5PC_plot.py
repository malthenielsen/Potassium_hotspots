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
    for i in range(36):
        #  plt.plot(data[0,:])
        #  plt.show()
        #  plt.plot(data[i,:])
        #  ax[i%3].plot(data[i,:])
        #  inds,_ = find_peaks(data[i,:], height = 10)
        inds,_ = find_peaks(data[i,:], height = -10)
        #  ax[i%3].scatter(inds, data[i,inds])
        length = np.max(np.where(~np.isnan(data[i,:]))[0])
        #  print(len(inds), length)
        

        if length > 20000:
            fps.append(len(inds)/(length*1/40)*1000)
        else:
            fps.append(np.nan)
    #  plt.show()
    fps = np.array(fps).reshape(-1,3)
    print(fps)
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
high = np.zeros((40,12))
medi = np.zeros((40,12))
lowi = np.zeros((40,12))

for i in range(40):
    path = f'dataL5PC/dataL5PC/soma_{i}.npy'
    fps = run(path)
    high[i,:] = fps[:,0]
    medi[i,:] = fps[:,1]
    lowi[i,:] = fps[:,2]
    #  for j in range(3):
        #  ax.plot(x_val, fps[:,j], color = colors[j], alpha = .5)


def sigmoid(x, xoff, b, MAX):
    return MAX* 1/(1 + np.exp(b*(x - xoff)))


#  angles = np.linspace(0,25,9)
#  angles = np.append(angles, np.array([40, 50, 75, 90]))
angles = np.linspace(0,25,9)
angles = np.append(angles, np.array([30, 50, 90]))

#  ax.plot(angles, sigmoid(angles, 40, 0.2, 0.04))
#  plt.show()

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
ax1.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing frequency [Hz]', title = 'L5PC neuron morphology')

p0, _ = curve_fit(sigmoid, angles, np.nanmean(high,axis=0), p0 = [20, 0.2, 10])
ax2.plot(con_angles, sigmoid(con_angles, *p0), color = colors[0])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(medi,axis=0), p0 = [20, 0.2, 10])
ax2.plot(con_angles, sigmoid(con_angles, *p0), color = colors[1])
p0, _ = curve_fit(sigmoid, angles, np.nanmean(lowi,axis=0), p0 = [20, 0.2, 10])
ax2.plot(con_angles, sigmoid(con_angles, *p0), color = colors[2])

ax2.errorbar(angles, np.nanmean(lowi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[2], label = 'High ')
ax2.scatter(angles, np.nanmean(lowi, axis = 0), color = colors[2])

ax2.errorbar(angles, np.nanmean(medi, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[1], label = 'Low ')
ax2.scatter(angles, np.nanmean(medi, axis = 0), color = colors[1])

ax2.errorbar(angles, np.nanmean(high, axis = 0), yerr = np.nanstd(fps, axis = 1)/np.sqrt(10), ls = '', capsize = 5, color = colors[0], label = 'None ')
ax2.scatter(angles, np.nanmean(high, axis = 0), color = colors[0])

#  ax.legend(title ='$\Delta E_K$')

#  ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing freq [AU]', title = 'L5PC Neuron')
#  ax2.set_xticks([0, 22.5, 45, 67.5, 90], [0, 22.5, 45, 67.5, 90])
ax2.legend(title = r'$\Delta E_{K^+}$ shift')
ax2.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing frequency [Hz]', title = 'L5PC neuron morphology')

ax1.set_xlim(-0.5,30)
ax2.set_xlim(80,90)
plt.tight_layout()
#  fig.savefig('Supp_weight_test', dpi = 200)=
#  fig.savefig('SuppF5.svg', dpi = 200)
fig.savefig('S8.pdf', dpi = 300)

plt.show()

