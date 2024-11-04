import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')


data = np.load('grid_local.npy')
x = data[:,-2]*2*np.pi
y = data[:,-1]*10
data = data[:,:-2][:,::100]
time = np.linspace(0,1000, data.shape[1])
print(data.shape)

from matplotlib.gridspec import GridSpec
fig = plt.figure( figsize = (13,5),constrained_layout = True)
gs = GridSpec(1, 5, figure=fig, )
ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1:])

ax1.set_xlabel('$2\pi r$')
ax1.set_ylabel('L')
ax2.set_ylabel('$\Delta [K^+]_o$ (mM)')
ax2.set_xlabel('Time (ms)')
ax1.set_xlim(0,2*np.pi)
ax1.set_ylim(0,10)
for i in range(10):
    ax1.scatter(x[i], y[i])
    ax2.plot(time, data[i,:])


ax1.set_title('Location of randomly selected points \n on the surface of the dendritic segment')
ax2.set_title('Measured $\Delta [K^+]_o$ at the randomly selected points')

#  fig.suptitle('Local changes in $\Delta [K^+]_o$ within the segment')
fig.savefig('Local_dKO_changes_for_rewiever_no_title.png', dpi = 200)
plt.show()


