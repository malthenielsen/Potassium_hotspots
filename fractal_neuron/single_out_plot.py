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
    #  fig, ax = plt.subplots(1,3, figsize = (12,4))
    for i in range(36):
        #  plt.plot(data[0,:])
        #  plt.show()
        #  plt.plot(data[i,:])
        #  ax[i%3].plot(data[i,:])
        inds,_ = find_peaks(data[i,:], height = 50)
        #  ax[i%3].scatter(inds, data[i,inds])
        length = np.max(np.where(~np.isnan(data[i,:]))[0])
        #  print(len(inds), length)
        

        if length > 20000:
            fps.append(len(inds)/(length*1/40)*1000)
        else:
            fps.append(0)
    #  plt.show()
    fps = np.array(fps).reshape(-1,3)
    #  plt.show()
    return fps


x_val = np.linspace(0,1,22)
fig, ax = plt.subplots(1,1, figsize = (8,6))
axt = ax.twinx()
#  colors = ['tab:green', 'tab:red', 'tab:blue']
colors = ['darkgrey', 'dodgerblue', 'goldenrod']
high = np.zeros((15,12))
medi = np.zeros((15,12))
lowi = np.zeros((15,12))

high2 = np.zeros((15,12))
medi2 = np.zeros((15,12))
lowi2 = np.zeros((15,12))

high3 = np.zeros((15,12))
medi3 = np.zeros((15,12))
lowi3 = np.zeros((15,12))

high4 = np.zeros((15,12))
medi4 = np.zeros((15,12))
lowi4 = np.zeros((15,12))

high5 = np.zeros((15,12))
medi5 = np.zeros((15,12))
lowi5 = np.zeros((15,12))

for i in range(15):
    path = f'data/SKv3/soma_{i}.npy'
    fps = run(path)
    high[i,:] = fps[:,0]
    medi[i,:] = fps[:,1]
    lowi[i,:] = fps[:,2]
    path = f'data/SKE2/soma_{i}.npy'
    fps = run(path)
    high2[i,:] = fps[:,0]
    medi2[i,:] = fps[:,1]
    lowi2[i,:] = fps[:,2]
    path = f'data/pas/soma_{i}.npy'
    fps = run(path)
    high3[i,:] = fps[:,0]
    medi3[i,:] = fps[:,1]
    lowi3[i,:] = fps[:,2]
    path = f'data/IM/soma_{i}.npy'
    fps = run(path)
    high4[i,:] = fps[:,0]
    medi4[i,:] = fps[:,1]
    lowi4[i,:] = fps[:,2]
    path = f'data/IH/soma_{i}.npy'
    fps = run(path)
    high5[i,:] = fps[:,0]
    medi5[i,:] = fps[:,1]
    lowi5[i,:] = fps[:,2]

#  exit()
def sigmoid(x, xoff, b, MAX):
    return MAX* 1/(1 + np.exp(b*(x - xoff)))


#  angles = np.linspace(0,35,9)
#  angles = np.append(angles, np.array([40, 50, 75, 90]))
angles = np.linspace(0,25,9)
angles = np.append(angles, np.array([30, 50, 90]))

#  ax.plot(angles, sigmoid(angles, 40, 0.2, 0.04))
#  plt.show()

con_angles = np.linspace(0,90,100)

#  ax.scatter(angles, np.nanmean(medi, axis = 0), color = colors[1])
#  ax.scatter(angles, np.nanmean(medi2, axis = 0), color = colors[1], marker = 'x')
ax.plot(angles, np.nanmean(medi, axis = 0))
ax.plot(angles, np.nanmean(medi2, axis = 0))
ax.plot(angles, np.nanmean(medi3, axis = 0))
ax.plot(angles, np.nanmean(medi4, axis = 0))
ax.plot(angles, np.nanmean(medi5, axis = 0))

axt.set_yticks([])

#  ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing freq [AU]', title = 'L5PC Neuron')
ax.set_xticks([0, 22.5, 45, 67.5, 90], [0, 22.5, 45, 67.5, 90])
ax.legend(['$K_{DR}$', '$K_{Ca}$', '$K_m$', 'Leak', '$K_{HCN}$' ], title ='Missing channel')
ax.set(xlabel = 'Tuning angle [deg]', ylabel = 'Firing frequency [Hz]', title = 'Fractal neuron morphology')
plt.tight_layout()
#  fig.savefig('Supp_IV_test', dpi = 200)
#  fig.savefig('SuppF5.svg', dpi = 200)

plt.show()



fig, ax = plt.subplots(1,1, figsize = (9,7))
high = np.nanmean(high, axis = 0)
high2 = np.nanmean(high2, axis = 0)
high3 = np.nanmean(high3, axis = 0)
high4 = np.nanmean(high4, axis = 0)
high5 = np.nanmean(high5, axis = 0)

medi = np.nanmean(medi, axis = 0)/high[0]
medi2 = np.nanmean(medi2, axis = 0)/high2[0]
medi3 = np.nanmean(medi3, axis = 0)/high3[0]
medi4 = np.nanmean(medi4, axis = 0)/high4[0]
medi5 = np.nanmean(medi5, axis = 0)/high5[0]

lowi = np.nanmean(lowi, axis = 0)/high[0]
lowi2 = np.nanmean(lowi2, axis = 0)/high2[0]
lowi3 = np.nanmean(lowi3, axis = 0)/high3[0]
lowi4 = np.nanmean(lowi4, axis = 0)/high4[0]
lowi5 = np.nanmean(lowi5, axis = 0)/high5[0]

high /= high[0]
high2 /= high2[0]
high3/= high3[0]
high4/= high4[0]
high5/= high5[0]

ax.scatter(0.1, high[0], color = colors[0])
ax.scatter(0.2, medi[0], color = colors[1])
ax.scatter(0.3, lowi[0], color = colors[2])

ax.scatter(1.1, high2[0], color = colors[0])
ax.scatter(1.2, medi2[0], color = colors[1])
ax.scatter(1.3, lowi2[0], color = colors[2])

ax.scatter(2.1, high3[0], color = colors[0])
ax.scatter(2.2, medi3[0], color = colors[1])
ax.scatter(2.3, lowi3[0], color = colors[2])

#  ax.scatter(3.1, high4[0], color = colors[0])
#  ax.scatter(3.2, medi4[0], color = colors[1])
#  ax.scatter(3.3, lowi4[0], color = colors[2])

ax.scatter(3.1, high5[0], color = colors[0])
ax.scatter(3.2, medi5[0], color = colors[1])
ax.scatter(3.3, lowi5[0], color = colors[2])

ax.legend(['None', 'Low', 'High'], title = r'$\Delta E_K$')

ax.set_xticks([0, 1, 2, 3],['$K_{DR}$', '$K_{Ca}$', '$K_m$', '$K_{HCN}$' ]) 
ax.set(xlabel = '$K^+$ channel muted', ylabel = r'Relative change in firing')
fig.savefig('Single_out_SI', dpi = 300)

plt.show()






