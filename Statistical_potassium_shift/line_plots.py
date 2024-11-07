import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')

cobalt = '#0047AB'
terra  = '#E3735E' 
tiffany = '#0ABAB5'


bin_1 = np.load('P_test_data/P_bin_10.npy')
bin_1 = np.nan_to_num(bin_1, 0)

bin_2 = np.load('P_test_data/P_bin_20.npy')
bin_3 = np.load('P_test_data/P_bin_30.npy')

def dKo(r2, dki):
    return dki / (r2**2 - 1)

fig, ax = plt.subplots(1,2, figsize = (10,6))
ori = np.linspace(0,90, len(bin_1))
print(ori)

ax[0].plot(ori, 2*bin_1*dKo(1.1, .2), color = plt.cm.Reds(.5))
ax[0].plot(ori, 2*bin_1*dKo(1.2, .2), color = plt.cm.Reds(.7))
ax[0].plot(ori, 2*bin_1*dKo(10.5, .2), color = plt.cm.Reds(.9))

ax[0].plot(ori, np.ones_like(bin_1)*dKo(1.1, .2) ,alpha = 1, color = plt.cm.Blues(.5)) 
ax[0].plot(ori, np.ones_like(bin_1)*dKo(1.2, .2) ,alpha = 1, color = plt.cm.Blues(.7))   
ax[0].plot(ori, np.ones_like(bin_1)*dKo(10.5, .2)  ,alpha = 1, color = plt.cm.Blues(.9))
#  ax[0].plot(ori, np.ones_like(bin_1)*dKo(1.1, .2),alpha = .5,   color = "FF0000", alpha = .5)
#  ax[0].plot(ori, np.ones_like(bin_1)*dKo(1.2, .2),alpha = .5,   color = "FF0000", alpha = .7)
#  ax[0].plot(ori, np.ones_like(bin_1)*dKo(1.5, .2),alpha = .5, color = "FF0000", alpha = .7)
ax[0].legend([1.1, 1.2, 3.5, 'Random', 'tmp', 'tmp'], title = '$R_2$')
ax[0].set_xlabel('$\Delta^\circ$ from target orientation')
ax[0].set_ylabel('$\Delta K_o$')
ax[0].set_title('Flux $K_o$, at diffrent $K_i$\n $R_2$ constant')

ax[1].plot(ori, 2*bin_1*dKo(np.sqrt(2), 1),    color = plt.cm.Reds(.5))
ax[1].plot(ori, 2*bin_1*dKo(np.sqrt(2), .5)  , color = plt.cm.Reds(.7))
ax[1].plot(ori, 2*bin_1*dKo(np.sqrt(2), .25),  color = plt.cm.Reds(.9))

ax[1].plot(ori, np.ones_like(bin_1)*dKo(np.sqrt(2), 1),   alpha = 1,   color = plt.cm.Blues(.5)) 
ax[1].plot(ori, np.ones_like(bin_1)*dKo(np.sqrt(2), .5),    alpha = 1, color = plt.cm.Blues(.7))   
ax[1].plot(ori, np.ones_like(bin_1)*dKo(np.sqrt(2), .25)  ,alpha = 1,  color = plt.cm.Blues(.9))
ax[1].legend([1, .5, .25, 'Random', 'tmp', 'tmp'], title = '$\Delta K_i$')
ax[1].set_xlabel('$\Delta^\circ$ from target orientation')
ax[1].set_ylabel('$\Delta K_o$')
ax[1].set_title('Flux $K_o$, at diffrent $R_2$\n $K_i$ constant')
ax[0].text(-0.1, 1.05, 'H', fontweight = 'bold', transform = ax[0].transAxes, fontsize = 20)
ax[0].set_xlim(-5,90)
ax[1].set_xlim(-5,90)
ax[0].set_ylim(-.3, 6)
ax[1].set_ylim(-.3, 6)

#  fig.savefig('Compare', dpi = 200)
#  fig.savefig('FIG_1H.svg', dpi = 400)
#  fig.savefig('FIG_1H.pdf', dpi = 400)
plt.show()

fig, ax = plt.subplots(figsize = (5,6))
ax.plot(ori, bin_1*2, color = 'black')
#  ax.plot(ori, bin_2, color = tiffany)
#  ax.plot(ori, bin_3, color = terra)
ax.legend(['$10^\circ$','$20^\circ$','$30^\circ$'], title = 'Spread in spine \n orientation distribution')
ax.set_ylabel(r'$Factor = \frac{\mathbf{E}(C)}{\mathbf{E}(R)}$')
ax.set_xlabel(r'$\Delta^{\circ}$ from target orientation')
ax.text(-0.1, 1.05, 'F', fontweight = 'bold', transform = ax.transAxes, fontsize = 20)
#  ax.set_xlim(-.5,90)
ax.set_ylim(-.3,6)
fig.savefig('Factor', dpi = 200)
fig.savefig('FIG_1F.svg', dpi = 400)
fig.savefig('FIG_1F.pdf', dpi = 400)


plt.show()




