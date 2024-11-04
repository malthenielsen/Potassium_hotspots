import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from scipy import stats


fig, ax = plt.subplots(figsize = (7,5))

x = np.linspace(0,100,1000)


ax.plot(x, stats.norm.cdf(x, 50, 15), color = 'red', label = r'$\theta$ = preferred orientation')
ax.plot(x, 0.6*stats.norm.cdf(x, 50, 15), color = 'black', label = r'$\theta \neq$ preferred orientation')
ax.set(xlabel = 'Time', ylabel = '$[K^+]_o$', title = 'Gaussian $K^+$ efflux')
ax.legend(title = r'Efflux from a synapse at orientation $\theta$')
ax.vlines(50, 0,1, color = 'darkgrey', ls = '--')
ax.text(35, 0.6, r'$T_{Peak}$', fontsize = 20)
ax.set_yticks([])
ax.set_xticks([])
plt.tight_layout()
fig.savefig('gauassian', dpi = 200)
plt.show()

