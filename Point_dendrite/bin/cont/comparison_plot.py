import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
import glob


spike = np.load('spike.npy', allow_pickle = True)
weight = np.load('weight.npy', allow_pickle = True)

fnames_w = glob.glob('80weight_*.npy')
np.random.shuffle(fnames_w)
fnames_s = glob.glob('spike_*.npy')



#  V_spike = spike.item().get('V')[::100][100:]
#  RT_spike = spike.item().get('RT')[::100][100:]
#
#  V_weight = weight.item().get('V')[::100][100:]
RT_weight = weight.item().get('RT')[::100][100:]


from matplotlib.gridspec import GridSpec
fig = plt.figure(layout="constrained", figsize = (14,4))
gs = GridSpec(9, 4, figure=fig)
ax = fig.add_subplot(gs[1:5, :3])
ax2 = fig.add_subplot(gs[5:, :3])
#  ax.plot(RT_spike, V_spike)
#  ax.plot(RT_weight, V_weight)
p_vec = np.zeros((2,9))
All_weight = 0
All_spike = 0
for i in range(0,20):
#  for i in range(10):
    V_weight = np.load(fnames_w[i])
    if i < 10:
        ax2.plot(V_weight, color = 'black', alpha = .5)
    All_weight += V_weight 
    V_weight = (V_weight[200:]).reshape(10,-1)

    V_spike = np.load(fnames_s[i])

    if i < 10:
        ax.plot(V_spike, color = 'red', alpha = .5)
    All_spike += V_spike 
    V_spike = (V_spike[200:]).reshape(10,-1)

    for j in range(9):
        if any(V_weight[j,:] > -30):
            p_vec[0,j] += 1
        if any(V_spike[j,:] > -30):
            p_vec[1,j] += 1
#  ax.plot(All_spike/20, color = 'red'   ,label = 'Frequency modulated')
#  ax2.plot(All_weight/20, color = 'black',label = 'Weight modulated')
#  ax.legend()



axu = fig.add_subplot(gs[0, :3])
axc = fig.add_subplot(gs[1:, -1])
r = np.linspace(-50,50)
theta = np.linspace(0,np.pi/4,9)
angles = np.arange(0,45,5)
for i in range(9):
    #  axu.vlines(200*i + 200, 1, -1)
    axu.plot(r*np.sin(theta[i]) + 200*i + 250, -r*np.cos(theta[i]), color = 'black')
    axu.text(200*i + 280, 0, str(angles[i]) + r'$^\circ$', fontsize = 15)
axu.text(100, 0, r'$\Delta\theta^\circ$', fontsize = 15)
axu.set_xlim(0, 2000)
ax.set_xlim(0, 2000)
ax2.set_xlim(0, 2000)
axu.set_xticks([])
axu.set_yticks([])
ax.set_xticks([])
#  ax.set_yticks([])
axu.axis('off')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('$V_m$ (mV)')
ax2.set_ylabel('$V_m$ (mV)')


#  fig, ax = plt.subplots(1,1, figsize = (7,5))
axc.errorbar(angles, p_vec[0,:]/20, yerr = np.sqrt(p_vec[0,:])/20, color = 'black', label = 'Weight modulated')
axc.errorbar(angles, p_vec[1,:]/20, yerr = np.sqrt(p_vec[1,:])/20, color = 'red', label = 'Frequency modulated')
axc.legend()
axc.set_xlabel(r'$\Delta \theta ^\circ$')
axc.set_ylabel('P firing')
axc.set_title('Firing probability')

fig.savefig('Comparison_method')
plt.show()

