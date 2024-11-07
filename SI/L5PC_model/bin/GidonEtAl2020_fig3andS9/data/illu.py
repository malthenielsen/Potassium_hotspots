import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from scipy.signal import find_peaks

angles = np.arange(0,50,5)
print(angles)

fig, ax = plt.subplots(1,1, figsize = (7,5))

linds = []
for a in angles:
    vvec = np.load(f'{a}.npy')
    inds,_ = find_peaks(vvec, height = 10)
    linds.append(len(inds))
    ax.plot(vvec)

fig, ax = plt.subplots(1,1, figsize = (7,5))
print(linds)
ax.scatter(angles, linds)
plt.show()

