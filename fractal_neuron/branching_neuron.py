import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
from create_segments_shai import *
h.load_file('stdrun.hoc')
h.nrn_load_dll('./mod_shai/x86_64/libnrnmech.so')
import subprocess


def change_ek(dend_list, clust_dend, ek, divisor):
    '''
    Here we change EK in the vicinity of the cluster. The clusters are always located in the middle of the segment,
    The changes might seem a bit extreeme, but that part is still work in progress. I Should not need to multiply the change by 10...
    '''
    for dend_idx in clust_dend:
        idx = 10
        dend_list[dend_idx-1].nseg = idx
        id_list = np.arange(0,idx, 1)
        np.random.shuffle(id_list)
        cluster = int(idx//divisor)
        mixed = int(idx//2)
        #  mixed = 0
        idx_cluster = id_list[:cluster]
        idx_mixed = id_list[cluster:cluster+mixed]
        for index, seg in enumerate(dend_list[dend_idx-1]):
            if index in idx_cluster:
                if ek != 0:
                    seg.SK_E2.dek += 1*ek
                    seg.SKv3_1.dek += 1*ek
                    seg.Im.dek += 1*ek
                    seg.e_pas += .6*ek
            elif index in idx_mixed:
                if ek != 0:
                    seg.SK_E2.dek += .7*ek
                    seg.SKv3_1.dek += .7*ek
                    seg.Im.dek += .7*ek
                    seg.e_pas += .5*ek
            else:
                if ek != 0:
                    seg.SK_E2.dek += .2*ek
                    seg.SKv3_1.dek += .2*ek
                    seg.Im.dek += .2*ek
                    seg.e_pas += .15*ek
    return dend_list


def create_neuron(tL, dek, stim_alpha, not_keep, test = 0, divisor = 1):
    '''
    tL is the trunk length

    dek is the change in EK

    stim_alpha is the orientation we are stimulating at, in the range from 0 to 90

    not_keep bool that determines if we are generating new cluster location and friings or not. This was added to better compare one simulation with another

    test is a parameter that either simulates many clusters when test = 0, or only one when test = 'single'
    '''
    if not_keep:
        subprocess.call(f"python fractal_prep.py {stim_alpha} {test}", shell=True)

    #Creating the neuron in the fractal manner
    #Namin as dend_XXY where XX is the segment number and Y is the level in the fractal

    dend_1  = create_dend(name = 'dend_010', L = 160)
    dend_2  = create_dend(name = 'dend_020', L = 160)
    dend_3  = create_dend(name = 'dend_031', L = 80)
    dend_4  = create_dend(name = 'dend_041', L = 80)
    dend_5  = create_dend(name = 'dend_051', L = 80)
    dend_6  = create_dend(name = 'dend_061', L = 80)
    dend_7  = create_dend(name = 'dend_072', L = 40)
    dend_8  = create_dend(name = 'dend_082', L = 40)
    dend_9  = create_dend(name = 'dend_092', L = 40)
    dend_10 = create_dend(name = 'dend_102', L = 40)
    dend_11 = create_dend(name = 'dend_112', L = 40)
    dend_12 = create_dend(name = 'dend_122', L = 40)
    dend_13 = create_dend(name = 'dend_132', L = 40)
    dend_14 = create_dend(name = 'dend_142', L = 40)
    dend_15 = create_dend(name = 'dend_153', L = 20)
    dend_16 = create_dend(name = 'dend_163', L = 20)
    dend_17 = create_dend(name = 'dend_173', L = 20)
    dend_18 = create_dend(name = 'dend_183', L = 20)
    dend_19 = create_dend(name = 'dend_193', L = 20)
    dend_20 = create_dend(name = 'dend_203', L = 20)
    dend_21 = create_dend(name = 'dend_213', L = 20)
    dend_22 = create_dend(name = 'dend_223', L = 20)
    dend_23 = create_dend(name = 'dend_233', L = 20)
    dend_24 = create_dend(name = 'dend_243', L = 20)
    dend_25 = create_dend(name = 'dend_253', L = 20)
    dend_26 = create_dend(name = 'dend_263', L = 20)
    dend_27 = create_dend(name = 'dend_273', L = 20)
    dend_28 = create_dend(name = 'dend_283', L = 20)
    dend_29 = create_dend(name = 'dend_293', L = 20)
    dend_30 = create_dend(name = 'dend_303', L = 20)

    trunk = create_trunk(name = 'trunk', L = tL)
    trunk.L = tL
    soma = create_soma(name = 'soma')


    dend_21.connect(dend_10, 1.0)
    dend_22.connect(dend_10, .95)
    dend_23.connect(dend_11, 1.0)
    dend_24.connect(dend_11, .95)
    dend_25.connect(dend_12, 1.0)
    dend_26.connect(dend_12, .95)
    dend_27.connect(dend_13, 1.0)
    dend_28.connect(dend_13, .95)
    dend_29.connect(dend_14, 1.0)
    dend_30.connect(dend_14, .95)

    dend_11.connect(dend_5, 1.0)
    dend_12.connect(dend_5, .95)
    dend_13.connect(dend_6, 1.0)
    dend_14.connect(dend_6, .95)
    dend_15.connect(dend_7, 1.0)
    dend_16.connect(dend_7, .95)
    dend_17.connect(dend_8, 1.0)
    dend_18.connect(dend_8, .95)
    dend_19.connect(dend_9, 1.0)
    dend_20.connect(dend_9, .95)

    dend_3.connect(dend_1 ,1.0)
    dend_4.connect(dend_1 ,.95)
    dend_5.connect(dend_2 ,1.0)
    dend_6.connect(dend_2 ,.95)
    dend_7.connect(dend_3 ,1.0)
    dend_8.connect(dend_3 ,.95)
    dend_9.connect(dend_4 ,1.0)
    dend_10.connect(dend_4, .95)

    dend_1.connect(trunk, 1.0)
    dend_2.connect(trunk, .95)

    trunk.connect(soma, 1.0)

    
    dend_list = [dend_1, dend_2, dend_3, dend_4, dend_5, dend_6, dend_7, dend_8,
                 dend_9, dend_10, dend_11, dend_12, dend_13, dend_14, dend_15,
                 dend_16, dend_17, dend_18, dend_19, dend_20, dend_21, dend_22, 
                 dend_23, dend_24, dend_25, dend_26, dend_27, dend_28, dend_29,
                 dend_30]

    '''
    Importing the files generated by fractal prep
    '''
    char_arr = np.load('CharFrac.npy') 
    StimVec = np.load('VecFrac.npy')
    clust_dend = np.load('dendrite_cluster_loc.npy')
    #  clust_dend = np.arange(15,30,1)
    #  clust_dend = np.array([30])
    
    '''
    Changes the EK in the segments where a cluster is located

    '''
    clust_dend = np.arange(0,30,1)
    #  clust_dend = [29, 13, 5, 1]

    dend_list = change_ek(dend_list, clust_dend, dek, divisor)

        

    synapses_list, netstims_list, netcons_list, rnd_list, vecT_list= [], [], [], [], []
    for i in range(char_arr.shape[0]):
        dend_n = int(char_arr[i,0])-1
        nmda_syn = h.nmda(dend_list[dend_n](char_arr[i,1]))
        #  nmda_syn.tau1 = 90
        ampa_syn =  h.AMPA(dend_list[dend_n](char_arr[i,1]))
        scalar = 50e-1

        '''
        Scalar is a value that is multiplied with the weight of the synapses.
        A way to adjust the model
        '''

        vecStim = h.VecStim()
        stim = StimVec[i,:]
        stim = np.delete(stim, np.where(stim == 0))

        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_ampa = h.NetCon(vecStim, ampa_syn)
        netCon_ampa.weight[0] = 1*scalar*(char_arr[i,-2])
        #  netCon_ampa.weight[0] = .2*scalar*weight
        netcons_list.append(netCon_ampa)
        synapses_list.append(ampa_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)


        vecStim = h.VecStim()
        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_nmda = h.NetCon(vecStim, nmda_syn)
        netCon_nmda.weight[0] =scalar*(char_arr[i,-2])
        #  netCon_nmda.weight[0] =scalar*weight
        netcons_list.append(netCon_nmda)
        synapses_list.append(nmda_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)

    '''
    Here I clamp the soma to simulate the mising random synapses
    '''
    #  iclamp = h.IClamp(soma(.5))
    #  iclamp.dur = 100
    #  iclamp.delay = 500
    #  #  iclamp.i = 1.5
    #  iclamp.amp = 2
    #
    #  iclamp2 = h.IClamp(soma(.5))
    #  iclamp2.dur = 100
    #  iclamp2.delay = 2500
    #  #  iclamp2.i = 1.5
    #  iclamp2.amp = 2

    t_vec = h.Vector().record(h._ref_t)
    V_vec = h.Vector().record(soma(0.0)._ref_v)
    D_vec = h.Vector().record(trunk(0.5)._ref_v)
    D_vec_1= h.Vector().record(dend_30(0.0)._ref_v)
    D_vec_2= h.Vector().record(dend_28(0.99)._ref_v)
    D_vec_3= h.Vector().record(dend_27(0.99)._ref_v)
    D_vec_4= h.Vector().record(dend_15(0.99)._ref_v)
    D_vec_5= h.Vector().record(dend_12(0.99)._ref_v)
    D_vec_6= h.Vector().record(dend_11(0.99)._ref_v)
    D_vec_7= h.Vector().record(dend_5(0.5)._ref_v)
    D_vec_8= h.Vector().record(dend_2(0.5)._ref_v)

    h.finitialize(-70)
    h.tstop = 2000
    h.run()
    stack = np.vstack([D_vec, D_vec_1,
              D_vec_2, D_vec_3,
              D_vec_4, D_vec_5,
              D_vec_6, D_vec_7,
              D_vec_8])
    return t_vec, V_vec, stack


'''
Here I test a single dendrite in blue vs multiple dendrites in green
'''

tl = np.arange(100,800,100)
tl = [100,300,700,1000]
dek = [0,10]
#  divisors = [1, 2, 3, 4, 5, 10]
#  for t in tl:
fig, ax = plt.subplots(1,1, figsize = (8,6))
fig_s, ax_s = plt.subplots(3,3, figsize = (12,10), sharex = True, sharey= True)
for i in range(20):
    stack_final = []
    soma_final = []
        #  print(div)
        #  if t == 100 and ek == 0:
            #  remix = 1
        #  else:
            #  remix = 0
    remix = 1
    #  print(remix)
    t_vec, V_vec, stack = create_neuron(400, 10, 0, remix, 'single', 4)
    stack_final.append(stack[1,:])
    soma_final.append(V_vec)
    #  np.save('stack_branch', stack)
    #  np.save(f'lunch_data/nLength={t}', V_vec)
    ax.plot(t_vec, V_vec)
    #  ax.plot(t_vec, stack[1,:])
    name = [30, 29, 28, 27, 15, 12, 11, 5, 2]
    ite = 0
    for i in range(3):
        for j in range(3):
            ax_s[i,j].plot(t_vec, stack[ite, :])
            ax_s[i,j].set_title(f'Dendrite {name[ite]}')
            #  ax_s[i,j].set_xlim(450,1100)
            ite += 1
#  break

    np.save(f'./probdata/stack_2{i}', np.vstack(stack_final))
    #  np.save(f'./probdata/soma_{i}', np.vstack(soma_final))
plt.show()



