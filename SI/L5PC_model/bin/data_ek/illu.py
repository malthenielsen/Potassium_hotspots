import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from scipy.signal import find_peaks

angles = np.arange(0,50,5)
angles = np.linspace(0,10,10)
#  angles = angles[[1,3,9]]
#  print(angles)


linds = []
fig, ax = plt.subplots(1,1, figsize = (7,5))
for a in angles:
    vvec = np.load(f'{a}.npy')
    inds,_ = find_peaks(vvec, height = 10)
    inds2 = np.where(~np.isnan(vvec))[0]
    length = np.max(inds2)/1000
    print(length, len(inds))
    linds.append(len(inds)/(length))
    ax.plot(vvec)
    #  ax.set_title(a)
    #  plt.show()

fig, ax = plt.subplots(1,1, figsize = (7,5))
print(linds)
ax.scatter(angles, linds)
plt.show()

