import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
from iminuit import Minuit

def run(path):
    colors = ['darkgrey', 'dodgerblue', 'goldenrod']
    angles = np.linspace(0,30,18)
    angles = np.append(angles, np.array([40, 50, 75, 90]))
    print(angles)
    data = np.load(path)
    data = data.reshape(22,3,-1)


    fig, ax = plt.subplots(3,3, figsize = (10, 8), sharex = True, sharey = True)
    index = [0, 6, 19]
    print(angles[index])
    for i in range(3):
        for j in range(3):
            ax[i,j].plot(data[index[i],j,5000:45000], color = colors[j])
    fig.savefig('4B.pdf', dpi = 400)

    plt.show()


#  path = './trunk_test/length_tuning/soma_1.npy'
for i in range(15):
    path = f'dataLOW/data/soma_{i}.npy'
    run(path)
