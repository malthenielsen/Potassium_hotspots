import numpy as np
from matplotlib import pyplot as plt
plt.style.use('science')

plt.style.use('K_PAPER')
from glob import glob
import os
from scipy import stats
from scipy.signal import find_peaks
from iminuit import Minuit



def func(path):
    fnames_soma = glob(f'{path}/soma?*.npy')
    fnames_soma.sort(key=os.path.getctime)
    data = np.zeros((3,22,len(fnames_soma)))


    for l, fname in enumerate(fnames_soma):
        print(fname)
        soma = np.load(fname)
        FPS_arr = np.zeros(22*3)
        v_arr = np.zeros(22*3)
        ISD = np.zeros((22*3, 100))

        bin_ = []
        binV = []
        #  print(len(soma), 'soma_length')
        #  print(fname)

        fig, ax = plt.subplots(1,1, figsize = (8,6))

        for i in range(len(soma)):
            ax.plot(soma[i])
        plt.show()

        #      inds, _ = find_peaks(soma[i], height = 50)
        #      FPS_arr[i]= len(inds)
        #      if len(inds) > 1:
        #          isd = np.diff(inds)
        #          ISD[i,:len(isd)] = isd
        #
        #      v_arr[i] = vinkel[i%22]
        #      #  plt.plot(soma[i])
        #      #  print(i)
        #  #  plt.show()
        #  #  exit()
        #      #  bin_.append(len(inds))
        #      #  binV.append(vinkel[i%23])
        #
        #
        #
        #  #  data[:,:,l] = np.array(bin_).reshape(23,3).T
        #  data[:,:,l] = FPS_arr.reshape(22,3).T
        #  dataV[:,:,l] = v_arr.reshape(3,22)
        #  #  dataV[:,:,l] = np.array(binV).reshape(23,3).T
        #  #  print(dataV[:,:,l])
        #  dataI[:,:,l,:] = ISD.reshape(22,3,100)

func('./')
