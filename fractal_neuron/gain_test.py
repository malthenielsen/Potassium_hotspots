import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
from create_segments_shai import *
h.load_file('stdrun.hoc')
h.nrn_load_dll('./mod_shai/x86_64/libnrnmech.so')
import subprocess
from scipy import stats
from multiprocessing import Pool
import time
from scipy.interpolate import CubicSpline


def tuning_vm(arr, s = 11, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(arr, kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)


def change_ek(dend_list, clust_dend, ek,mix_ek):
    '''
    Here we change EK in the vicinity of the cluster. The clusters are always located in the middle of the segment,
    The changes might seem a bit extreeme, but that part is still work in progress. I Should not need to multiply the change by 10...
    '''
    for dend_idx in clust_dend:
        idx = 10
        dend_list[dend_idx-1].nseg = idx
        id_list = np.arange(0,idx, 1)
        np.random.shuffle(id_list)
        cluster = int(3*idx//4)
        mixed = 0#kjint(idx//4)
        #  mixed = 0
        idx_cluster = id_list[:cluster]
        for index, seg in enumerate(dend_list[dend_idx-1]):
            if index in idx_cluster:
                if ek != 0:
                    seg.SK_E2.dek += 1*ek
                    seg.SKv3_1.dek += 1*ek
                    seg.Im.dek += 1*ek
                    seg.e_pas += .6*ek
            else:
                if ek != 0:
                    seg.SK_E2.dek += 2
                    seg.SKv3_1.dek += 2
                    seg.Im.dek += 2
                    seg.e_pas += 2
    return dend_list


def create_neuron(tL, dek, mix_dek, stim_alpha, not_keep, test = 0, ID = 1):
    '''
    tL is the trunk length

    dek is the change in EK

    stim_alpha is the orientation we are stimulating at, in the range from 0 to 90

    not_keep bool that determines if we are generating new cluster location and friings or not. This was added to better compare one simulation with another

    test is a parameter that either simulates many clusters when test = 0, or only one when test = 'single'

    '''
    if not_keep:
        #  print('made new')
        subprocess.call(f"python3 3fractal_prep.py {stim_alpha} {test} {ID}", shell=True)

    char_arr = np.load(f'bin/3CharFrac_{ID}.npy') 
    StimVec = np.load(f'bin/3VecFrac_{ID}.npy')

    #Creating the neuron in the fractal manner
    #Namin as dend_XXY where XX is the segment number and Y is the level in the fractal

    dend_1  = create_dend(name = 'dend_1', L = 160)
    dend_2  = create_dend(name = 'dend_2', L = 160)
    dend_3  = create_dend(name = 'dend_3', L = 160)

    dend_11  = create_dend(name = 'dend_11', L = 80)
    dend_12  = create_dend(name = 'dend_12', L = 80)
    dend_13  = create_dend(name = 'dend_13', L = 80)
    dend_21  = create_dend(name = 'dend_21', L = 80)
    dend_22  = create_dend(name = 'dend_22', L = 80)
    dend_23  = create_dend(name = 'dend_23', L = 80)
    dend_31  = create_dend(name = 'dend_31', L = 80)
    dend_32  = create_dend(name = 'dend_32', L = 80)
    dend_33  = create_dend(name = 'dend_33', L = 80)

    dend_111 = create_dend(name = 'dend_111', L = 40)
    dend_121 = create_dend(name = 'dend_121', L = 40)
    dend_131 = create_dend(name = 'dend_131', L = 40)
    dend_211 = create_dend(name = 'dend_211', L = 40)
    dend_221 = create_dend(name = 'dend_221', L = 40)
    dend_231 = create_dend(name = 'dend_231', L = 40)
    dend_311 = create_dend(name = 'dend_311', L = 40)
    dend_321 = create_dend(name = 'dend_321', L = 40)
    dend_331 = create_dend(name = 'dend_331', L = 40)
    dend_112 = create_dend(name = 'dend_112', L = 40)
    dend_122 = create_dend(name = 'dend_122', L = 40)
    dend_132 = create_dend(name = 'dend_132', L = 40)
    dend_212 = create_dend(name = 'dend_212', L = 40)
    dend_222 = create_dend(name = 'dend_222', L = 40)
    dend_232 = create_dend(name = 'dend_232', L = 40)
    dend_312 = create_dend(name = 'dend_312', L = 40)
    dend_322 = create_dend(name = 'dend_322', L = 40)
    dend_332 = create_dend(name = 'dend_332', L = 40)
    dend_113 = create_dend(name = 'dend_113', L = 40)
    dend_123 = create_dend(name = 'dend_123', L = 40)
    dend_133 = create_dend(name = 'dend_133', L = 40)
    dend_213 = create_dend(name = 'dend_213', L = 40)
    dend_223 = create_dend(name = 'dend_223', L = 40)
    dend_233 = create_dend(name = 'dend_233', L = 40)
    dend_313 = create_dend(name = 'dend_313', L = 40)
    dend_323 = create_dend(name = 'dend_323', L = 40)
    dend_333 = create_dend(name = 'dend_333', L = 40)

    dend_1111 = create_dend(name = 'dend_1111', L = 20)
    dend_1211 = create_dend(name = 'dend_1211', L = 20)
    dend_1311 = create_dend(name = 'dend_1311', L = 20)
    dend_2111 = create_dend(name = 'dend_2111', L = 20)
    dend_2211 = create_dend(name = 'dend_2211', L = 20)
    dend_2311 = create_dend(name = 'dend_2311', L = 20)
    dend_3111 = create_dend(name = 'dend_3111', L = 20)
    dend_3211 = create_dend(name = 'dend_3211', L = 20)
    dend_3311 = create_dend(name = 'dend_3311', L = 20)
    dend_1121 = create_dend(name = 'dend_1121', L = 20)
    dend_1221 = create_dend(name = 'dend_1221', L = 20)
    dend_1321 = create_dend(name = 'dend_1321', L = 20)
    dend_2121 = create_dend(name = 'dend_2121', L = 20)
    dend_2221 = create_dend(name = 'dend_2221', L = 20)
    dend_2321 = create_dend(name = 'dend_2321', L = 20)
    dend_3121 = create_dend(name = 'dend_3121', L = 20)
    dend_3221 = create_dend(name = 'dend_3221', L = 20)
    dend_3321 = create_dend(name = 'dend_3321', L = 20)
    dend_1131 = create_dend(name = 'dend_1131', L = 20)
    dend_1231 = create_dend(name = 'dend_1231', L = 20)
    dend_1331 = create_dend(name = 'dend_1331', L = 20)
    dend_2131 = create_dend(name = 'dend_2131', L = 20)
    dend_2231 = create_dend(name = 'dend_2231', L = 20)
    dend_2331 = create_dend(name = 'dend_2331', L = 20)
    dend_3131 = create_dend(name = 'dend_3131', L = 20)
    dend_3231 = create_dend(name = 'dend_3231', L = 20)
    dend_3331 = create_dend(name = 'dend_3331', L = 20)
    dend_1112 = create_dend(name = 'dend_1112', L = 20)
    dend_1212 = create_dend(name = 'dend_1212', L = 20)
    dend_1312 = create_dend(name = 'dend_1312', L = 20)
    dend_2112 = create_dend(name = 'dend_2112', L = 20)
    dend_2212 = create_dend(name = 'dend_2212', L = 20)
    dend_2312 = create_dend(name = 'dend_2312', L = 20)
    dend_3112 = create_dend(name = 'dend_3112', L = 20)
    dend_3212 = create_dend(name = 'dend_3212', L = 20)
    dend_3312 = create_dend(name = 'dend_3312', L = 20)
    dend_1122 = create_dend(name = 'dend_1122', L = 20)
    dend_1222 = create_dend(name = 'dend_1222', L = 20)
    dend_1322 = create_dend(name = 'dend_1322', L = 20)
    dend_2122 = create_dend(name = 'dend_2122', L = 20)
    dend_2222 = create_dend(name = 'dend_2222', L = 20)
    dend_2322 = create_dend(name = 'dend_2322', L = 20)
    dend_3122 = create_dend(name = 'dend_3122', L = 20)
    dend_3222 = create_dend(name = 'dend_3222', L = 20)
    dend_3322 = create_dend(name = 'dend_3322', L = 20)
    dend_1132 = create_dend(name = 'dend_1132', L = 20)
    dend_1232 = create_dend(name = 'dend_1232', L = 20)
    dend_1332 = create_dend(name = 'dend_1332', L = 20)
    dend_2132 = create_dend(name = 'dend_2132', L = 20)
    dend_2232 = create_dend(name = 'dend_2232', L = 20)
    dend_2332 = create_dend(name = 'dend_2332', L = 20)
    dend_3132 = create_dend(name = 'dend_3132', L = 20)
    dend_3232 = create_dend(name = 'dend_3232', L = 20)
    dend_3332 = create_dend(name = 'dend_3332', L = 20)
    dend_1113 = create_dend(name = 'dend_1113', L = 20)
    dend_1213 = create_dend(name = 'dend_1213', L = 20)
    dend_1313 = create_dend(name = 'dend_1313', L = 20)
    dend_2113 = create_dend(name = 'dend_2113', L = 20)
    dend_2213 = create_dend(name = 'dend_2213', L = 20)
    dend_2313 = create_dend(name = 'dend_2313', L = 20)
    dend_3113 = create_dend(name = 'dend_3113', L = 20)
    dend_3213 = create_dend(name = 'dend_3213', L = 20)
    dend_3313 = create_dend(name = 'dend_3313', L = 20)
    dend_1123 = create_dend(name = 'dend_1123', L = 20)
    dend_1223 = create_dend(name = 'dend_1223', L = 20)
    dend_1323 = create_dend(name = 'dend_1323', L = 20)
    dend_2123 = create_dend(name = 'dend_2123', L = 20)
    dend_2223 = create_dend(name = 'dend_2223', L = 20)
    dend_2323 = create_dend(name = 'dend_2323', L = 20)
    dend_3123 = create_dend(name = 'dend_3123', L = 20)
    dend_3223 = create_dend(name = 'dend_3223', L = 20)
    dend_3323 = create_dend(name = 'dend_3323', L = 20)
    dend_1133 = create_dend(name = 'dend_1133', L = 20)
    dend_1233 = create_dend(name = 'dend_1233', L = 20)
    dend_1333 = create_dend(name = 'dend_1333', L = 20)
    dend_2133 = create_dend(name = 'dend_2133', L = 20)
    dend_2233 = create_dend(name = 'dend_2233', L = 20)
    dend_2333 = create_dend(name = 'dend_2333', L = 20)
    dend_3133 = create_dend(name = 'dend_3133', L = 20)
    dend_3233 = create_dend(name = 'dend_3233', L = 20)
    dend_3333 = create_dend(name = 'dend_3333', L = 20)
    

    trunk = create_trunk(name = 'trunk', L = tL)
    trunk.L = tL
    soma = create_soma(name = 'soma')

    dend_1111.connect(dend_111, 1.0)
    dend_1112.connect(dend_111, .99)
    dend_1113.connect(dend_111, .98)
    dend_1121.connect(dend_112, 1.0)
    dend_1122.connect(dend_112, .99)
    dend_1123.connect(dend_112, .98)
    dend_1131.connect(dend_113, 1.0)
    dend_1132.connect(dend_113, .99)
    dend_1133.connect(dend_113, .98)
    dend_1211.connect(dend_121, 1.0)
    dend_1212.connect(dend_121, .99)
    dend_1213.connect(dend_121, .98)
    dend_1221.connect(dend_122, 1.0)
    dend_1222.connect(dend_122, .99)
    dend_1223.connect(dend_122, .98)
    dend_1231.connect(dend_123, 1.0)
    dend_1232.connect(dend_123, .99)
    dend_1233.connect(dend_123, .98)
    dend_1311.connect(dend_131, 1.0)
    dend_1312.connect(dend_131, .99)
    dend_1313.connect(dend_131, .98)
    dend_1321.connect(dend_132, 1.0)
    dend_1322.connect(dend_132, .99)
    dend_1323.connect(dend_132, .98)
    dend_1331.connect(dend_133, 1.0)
    dend_1332.connect(dend_133, .99)
    dend_1333.connect(dend_133, .98)
    dend_2111.connect(dend_211, 1.0)
    dend_2112.connect(dend_211, .99)
    dend_2113.connect(dend_211, .98)
    dend_2121.connect(dend_212, 1.0)
    dend_2122.connect(dend_212, .99)
    dend_2123.connect(dend_212, .98)
    dend_2131.connect(dend_213, 1.0)
    dend_2132.connect(dend_213, .99)
    dend_2133.connect(dend_213, .98)
    dend_2211.connect(dend_221, 1.0)
    dend_2212.connect(dend_221, .99)
    dend_2213.connect(dend_221, .98)
    dend_2221.connect(dend_222, 1.0)
    dend_2222.connect(dend_222, .99)
    dend_2223.connect(dend_222, .98)
    dend_2231.connect(dend_223, 1.0)
    dend_2232.connect(dend_223, .99)
    dend_2233.connect(dend_223, .98)
    dend_2311.connect(dend_231, 1.0)
    dend_2312.connect(dend_231, .99)
    dend_2313.connect(dend_231, .98)
    dend_2321.connect(dend_232, 1.0)
    dend_2322.connect(dend_232, .99)
    dend_2323.connect(dend_232, .98)
    dend_2331.connect(dend_233, 1.0)
    dend_2332.connect(dend_233, .99)
    dend_2333.connect(dend_233, .98)
    dend_3111.connect(dend_311, 1.0)
    dend_3112.connect(dend_311, .99)
    dend_3113.connect(dend_311, .98)
    dend_3121.connect(dend_312, 1.0)
    dend_3122.connect(dend_312, .99)
    dend_3123.connect(dend_312, .98)
    dend_3131.connect(dend_313, 1.0)
    dend_3132.connect(dend_313, .99)
    dend_3133.connect(dend_313, .98)
    dend_3211.connect(dend_321, 1.0)
    dend_3212.connect(dend_321, .99)
    dend_3213.connect(dend_321, .98)
    dend_3221.connect(dend_322, 1.0)
    dend_3222.connect(dend_322, .99)
    dend_3223.connect(dend_322, .98)
    dend_3231.connect(dend_323, 1.0)
    dend_3232.connect(dend_323, .99)
    dend_3233.connect(dend_323, .98)
    dend_3311.connect(dend_331, 1.0)
    dend_3312.connect(dend_331, .99)
    dend_3313.connect(dend_331, .98)
    dend_3321.connect(dend_332, 1.0)
    dend_3322.connect(dend_332, .99)
    dend_3323.connect(dend_332, .98)
    dend_3331.connect(dend_333, 1.0)
    dend_3332.connect(dend_333, .99)
    dend_3333.connect(dend_333, .98)

    dend_111.connect(dend_11, 1.0)
    dend_112.connect(dend_11, .99)
    dend_113.connect(dend_11, .98)
    dend_121.connect(dend_12, 1.0)
    dend_122.connect(dend_12, .99)
    dend_123.connect(dend_12, .98)
    dend_131.connect(dend_13, 1.0)
    dend_132.connect(dend_13, .99)
    dend_133.connect(dend_13, .98)
    dend_211.connect(dend_21, 1.0)
    dend_212.connect(dend_21, .99)
    dend_213.connect(dend_21, .98)
    dend_221.connect(dend_22, 1.0)
    dend_222.connect(dend_22, .99)
    dend_223.connect(dend_22, .98)
    dend_231.connect(dend_23, 1.0)
    dend_232.connect(dend_23, .99)
    dend_233.connect(dend_23, .98)
    dend_311.connect(dend_31, 1.0)
    dend_312.connect(dend_31, .99)
    dend_313.connect(dend_31, .98)
    dend_321.connect(dend_32, 1.0)
    dend_322.connect(dend_32, .99)
    dend_323.connect(dend_32, .98)
    dend_331.connect(dend_33, 1.0)
    dend_332.connect(dend_33, .99)
    dend_333.connect(dend_33, .98)

    dend_11.connect(dend_1, 1.0)
    dend_12.connect(dend_1, .99)
    dend_13.connect(dend_1, .98)
    dend_21.connect(dend_2, 1.0)
    dend_22.connect(dend_2, .99)
    dend_23.connect(dend_2, .98)
    dend_31.connect(dend_3, 1.0)
    dend_32.connect(dend_3, .99)
    dend_33.connect(dend_3, .98)

    dend_1.connect(trunk, 1.0)
    dend_2.connect(trunk, .99)
    dend_3.connect(trunk, .98)


    trunk.connect(soma, 1.0)
    

    
    #  dend_list = [dend_1, dend_2, dend_3, dend_4, dend_5, dend_6, dend_7, dend_8,
    #               dend_9, dend_10, dend_11, dend_12, dend_13, dend_14, dend_15,
    #               dend_16, dend_17, dend_18, dend_19, dend_20, dend_21, dend_22,
    #               dend_23, dend_24, dend_25, dend_26, dend_27, dend_28, dend_29,
    #               dend_30]
    dend_list = [dend_1, dend_2, dend_3, dend_11, dend_12, dend_13, dend_21, dend_22, dend_23, dend_31, dend_32, dend_33, dend_111, dend_112, dend_113, dend_121, dend_122, dend_123, dend_131, dend_132, dend_133, dend_211, dend_212, dend_213, dend_221, dend_222, dend_223, dend_231, dend_232, dend_233, dend_311, dend_312, dend_313, dend_321, dend_322, dend_323, dend_331, dend_332, dend_333,dend_1111, dend_1112, dend_1113, dend_1121, dend_1122, dend_1123, dend_1131, dend_1132, dend_1133, dend_1211, dend_1212, dend_1213, dend_1221, dend_1222, dend_1223, dend_1231, dend_1232, dend_1233, dend_1311, dend_1312, dend_1313, dend_1321, dend_1322, dend_1323, dend_1331, dend_1332, dend_1333, dend_2111, dend_2112, dend_2113, dend_2121, dend_2122, dend_2123, dend_2131, dend_2132, dend_2133, dend_2211, dend_2212, dend_2213, dend_2221, dend_2222, dend_2223, dend_2231, dend_2232, dend_2233, dend_2311, dend_2312, dend_2313, dend_2321, dend_2322, dend_2323, dend_2331, dend_2332, dend_2333, dend_3111, dend_3112, dend_3113, dend_3121, dend_3122, dend_3123, dend_3131, dend_3132, dend_3133, dend_3211, dend_3212, dend_3213, dend_3221, dend_3222, dend_3223, dend_3231, dend_3232, dend_3233, dend_3311, dend_3312, dend_3313, dend_3321, dend_3322, dend_3323, dend_3331, dend_3332, dend_3333]

    print(len(dend_list))

    '''
    Importing the files generated by fractal prep
    '''
    char_arr = np.load(f'bin/3CharFrac_{ID}.npy') 
    StimVec = np.load(f'bin/3VecFrac_{ID}.npy')
    #  clust_dend = np.load('3dendrite_cluster_loc.npy')
#  clust_dend = np.arange(15,30,1)
#  clust_dend = np.array([30])
    
    '''
    Changes the EK in the segments where a cluster is located

    '''
    clust_dend = np.arange(0,120,1)
    #  clust_dend = [29, 13, 5, 1]

    dend_list = change_ek(dend_list, clust_dend, dek, mix_dek)

        

    synapses_list, netstims_list, netcons_list, rnd_list, vecT_list= [], [], [], [], []
    for i in range(char_arr.shape[0]):
        dend_n = int(char_arr[i,0])-1
        nmda_syn = h.nmda(dend_list[dend_n](char_arr[i,1]))
        #  nmda_syn.tau1 = 90
        ampa_syn =  h.AMPA(dend_list[dend_n](char_arr[i,1]))
        #  noise = h.Gfluct(dend_list[dend_n](char_arr[i,1]))
        if dek != 0:
            nmda_syn.e = 1

        #scalar = 50e-1
        scalar = 50e-1

        '''
        Scalar is a value that is multiplied with the weight of the synapses.
        A way to adjust the model
        '''

        vecStim = h.VecStim()
        stim = StimVec[i,:]
        stim = np.delete(stim, np.where(stim == 0))
        weight = tuning_vm(char_arr[i,-1], 11, stim_alpha)
        if char_arr[i, -1] == 100:
            weight = 0

        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_ampa = h.NetCon(vecStim, ampa_syn)
        #netCon_ampa.weight[0] = 1*scalar*(char_arr[i,-2])
        netCon_ampa.weight[0] = 1*scalar*weight
        netcons_list.append(netCon_ampa)
        synapses_list.append(ampa_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)


        vecStim = h.VecStim()
        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_nmda = h.NetCon(vecStim, nmda_syn)
        netCon_nmda.weight[0] =scalar*weight
        netcons_list.append(netCon_nmda)
        synapses_list.append(nmda_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)

        #  vecStim = h.VecStim()
        #  vec = h.Vector(np.sort(stim.astype(int)))
        #  netCon_noise = h.NetCon(vecStim, noise)
        #  netCon_noise.weight[0] = 0*.25
        #  netcons_list.append(netCon_noise)
        #  synapses_list.append(noise)
        #  vecT_list.append(vecStim)
        #  vecStim.play(vec)

    '''
    Here I clamp the soma to simulate the mising random synapses
    '''
    t_vec = h.Vector().record(h._ref_t)
    V_vec = h.Vector().record(soma(0.0)._ref_v)
    D_vec = h.Vector().record(trunk(0.5)._ref_v)

    h.finitialize(-70)
    h.tstop = 2000
    h.run()
    #  stack = np.vstack([D_vec, D_vec_1,
    #            D_vec_2, D_vec_3,
    #            D_vec_4, D_vec_5,
    #            D_vec_6, D_vec_7,
    return t_vec, V_vec, D_vec


'''
Here I test a single dendrite in blue vs multiple dendrites in green
'''

tl = [500]
dek = [0,6,16]
P_dek = np.load('P_bin_10.npy')
P_dek = np.nan_to_num(P_dek)
p1d = CubicSpline(angles, P_dek/np.nanmax(P_dek))


def run(i):
    stack_final = []
    soma_final = []
    t = tl[0]
    time.sleep(i)
    ID = int(time.time())
    angles = np.random.uniform(0,90, 20)
    for idx, a in enumerate(angles):
        if idx == 0:
            remix = True
        else:
            remix = 0
        for ek in dek:
            #  create_neuron(t, dek_a(a)*ek, dek_a_mix(a)*ek, a, remix, 'ingle', ID)
            t_vec, V_vec, stack = create_neuron(t, p1d(a)*ek, p1d(a)*ek, a, remix, 'ingle', ID)
            soma_final.append(V_vec)
            print(ek, a, i)

    np.save(f'./data/gain/angle_{i}', np.vstack(angles))
    np.save(f'./data/gain/soma_{i}', np.vstack(soma_final))

if __name__ == '__main__':
    index = np.arange(0,50,1)
    pool = Pool(50)
    pool.map(run, index)
    pool.close()
    pool.join()


