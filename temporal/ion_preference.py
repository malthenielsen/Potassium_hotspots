import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')


plt.rcParams.update({'font.size': 14})
plt.rcParams['svg.fonttype'] = 'none'


def kion(ki,ko, delta, dem = 10):
    return np.log((ko + delta)/(ki - delta/dem)) - np.log(ko/ki)

def naion(nai,nao, delta, dem = 10):
    return abs(np.log((nao - delta)/(nai + delta/dem)) - np.log(nao/nai))

def caion(cai,cao, delta, dem = 10):
    return .5*np.abs(np.log((cao - delta)/(cai + delta/dem)) - np.log(cao/cai))

ki = 120
ko = 4
nai = 7
nao = 140
cao = 1.5
cai = 0.0001


delta = np.linspace(0,15,1000)
fig, ax = plt.subplots(1,3, figsize = (15,6))
ax[0].plot(delta, kion(ki, ko, delta), label = '$K^+$', color = plt.cm.inferno_r(10/12))
ax[0].plot(delta, naion(nai, nao, delta), label = '$Na^+$', color = plt.cm.inferno_r(10/12), ls = '--')
ax[0].plot(delta, kion(ki, ko, delta, dem = 5), color = plt.cm.inferno_r(5/12))
ax[0].plot(delta, naion(nai, nao, delta, dem = 5), color = plt.cm.inferno_r(5/12), ls = '--')
ax[0].plot(delta, kion(ki, ko, delta, dem = 2), color = plt.cm.inferno_r(2/12))
ax[0].plot(delta, naion(nai, nao, delta, dem = 2), color = plt.cm.inferno_r(2/12), ls = '--')
ax[0].set(xlabel = '$\Delta[ion](mMol)$', ylabel = 'Absolute change in potential [AU]', title = '$\Delta E_{ion}$ changes') 
ax[0].legend()
ax[1].plot(delta, kion(ki, ko, delta)/naion(nai,nao, delta)                  , color = plt.cm.inferno_r(10/12))
ax[1].plot(delta, kion(ki, ko, delta, dem = 5)/naion(nai,nao, delta, dem = 5), color = plt.cm.inferno_r(5/12))
ax[1].plot(delta, kion(ki, ko, delta, dem = 2)/naion(nai,nao, delta, dem = 2), color = plt.cm.inferno_r(2/12))
ax[1].set(xlabel = '$\Delta[ion](mMol)$', ylabel = 'Ratio', title = r'Ratio between $\frac{\Delta E_K}{\Delta E_{Na}}$') 
ax[1].legend([10, 5, 2], title = r'$V_R = \frac{V_{in}}{V_{out}}$')

delta = np.linspace(0,0.001,1000)
deltap = np.linspace(0,1,1000)
#  fig, ax = plt.subplots(1,1, figsize = (7,5))
ax[2].plot(deltap, caion(cai, cao, delta), label = '$Ca^{2+}$', color = plt.cm.inferno_r(10/12), ls = ':')
ax[2].plot(deltap, caion(cai, cao, delta, dem = 5), color = plt.cm.inferno_r(5/12), ls = ':')
ax[2].plot(deltap, caion(cai, cao, delta, dem = 2), color = plt.cm.inferno_r(2/12), ls = ':')
ax[2].set(xlabel = '$\Delta[ion](\mu Mol)$', ylabel = 'Absolute change in potential [AU]', title = '$\Delta E_{ion}$ changes') 
ax[2].legend()
ax[0].set_ylim(0,1.8)
ax[2].set_ylim(0,1.8)
fig.savefig('ion_ratio')
fig.savefig('ion_ratio.svg')

plt.show()


