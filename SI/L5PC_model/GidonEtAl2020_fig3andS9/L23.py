import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from neuron import h, gui
#  h.nrn_load_dll('../mod/x86_64/libnrnmech.so')
h.nrn_load_dll('/home/nordentoft/Documents/Potassium_and_dendrites/fractal_neuron/mod_shai/x86_64/libnrnmech.so')
h.load_file('stdrun.hoc')
import subprocess
from scipy import stats, interpolate
import time
from multiprocessing import Pool

def tuning_vm(arr, s = 11, shift = 0):
    shift = np.radians(shift)*2 #von mises is for the full circle we only look at the half
    kappa = 1/np.power(np.deg2rad(s)*2, 2) # relation from kappa to std is std**2 = 1/k
    arr_r = np.linspace(-np.pi,np.pi, 100)
    val = stats.vonmises.pdf(arr, kappa, loc = 0 + shift)
    val_r = stats.vonmises.pdf(arr_r, kappa, loc = 0 + shift)
    return val / np.max(val_r)


def neuron(ek):
    morphology_file = '_morph/171122005.hoc'
    m = h.load_file(morphology_file)
#  print(dir(h.load_file(morphology_file)))
    m = h.allsec()

#  for sec in m:
        #  print(sec)


    for dend in m.dend:
        #  dend.insert('Ih')
        dend.insert('pas')
        #  dend.gIhbar_Ih = 0.0001/2
        dend.cm = 2
        dend.g_pas =6e-5



    for apic in m.apic:
        apic.insert('CaDynamics_E2')
        apic.insert('SK_E2')
        apic.insert('Ca_LVAst')
        apic.insert('Ca_HVA')
        apic.insert('SKv3_1')
        apic.insert('NaTs2_t')
        apic.insert('Im')
        #  apic.insert('Ih')
        apic.insert('pas')
        apic.ek =-85
        apic.ena = 50
        apic.cm = 2
        apic.g_pas = 6e-5
        apic.decay_CaDynamics_E2 = 35.725651
        apic.gamma_CaDynamics_E2 = 0.000637
        apic.gSK_E2bar_SK_E2 = 0.000002
        apic.gCa_HVAbar_Ca_HVA = 0.000701
        apic.gSKv3_1bar_SKv3_1 = 0.001808
        apic.gNaTs2_tbar_NaTs2_t = 0.021489
        apic.gImbar_Im = 0.00099
        #  apic.gIhbar_Ih =  .00015 //0.00001*1.5

    for al in m.all:
        al.insert('pas')
        al.cm = 2
        al.Ra = 100
        al.e_pas = -90
        al.nseg = 5

    m.soma.insert('Im')
    m.soma.insert('Ca_LVAst')
    m.soma.insert('Ca_HVA')
    m.soma.insert('CaDynamics_E2')
    m.soma.insert('SK_E2')
    m.soma.insert('SKv3_1')
    m.soma.insert('NaTs2_t')
    m.soma.insert('pas')
    m.soma.ek = -85
    m.soma.ena = 50
    m.soma.insert('Ih')
    m.soma.gIhbar_Ih = 0.0001*0.75
    m.soma.g_pas = 3e-6
    m.soma.gImbar_Im = 0.000008*1
    m.soma.decay_CaDynamics_E2 = 294.6795
    m.soma.gamma_CaDynamics_E2 = 0.000557
    m.soma.gCa_LVAstbar_Ca_LVAst = 0.000557
    m.soma.gCa_HVAbar_Ca_HVA = 0.000644
    m.soma.gSK_E2bar_SK_E2 = 0.09965*1
    m.soma.gSKv3_1bar_SKv3_1 = 0.338029*1
    m.soma.gNaTs2_tbar_NaTs2_t = 0.998912*5


    not_keep = 0
    stim_alpha = 15
    test = 'none'
    ID = 1
    dek = 1
    #  ek = 10
    mix_ek = 2

    if not_keep:
        print('made new')
        subprocess.call(f"python3 L23_prep_test.py {stim_alpha} {test} {ID}", shell=True)

    char_arr = np.load(f'bin/3CharFrac_{ID}.npy') 
    StimVec = np.load(f'bin/3VecFrac_{ID}.npy')

    idx = np.where(char_arr[:,2] != 0)
    char_arr = char_arr[idx[0],:]
    StimVec = StimVec[idx[0],:]
    for apic in m.apic:
        idx = 10
        id_list = np.arange(0,idx, 1)
        np.random.shuffle(id_list)
        cluster = int(2*idx//4) 
        mixed = 0
        idx_cluster = id_list[:cluster]
        for index, seg in enumerate(apic):
            if index in idx_cluster:
                if ek != 0:
                    seg.SK_E2.dek += 1*ek
                    seg.SKv3_1.dek += 1*ek
                    seg.Im.dek += 1*ek
                    seg.e_pas += .6*ek
            else:
                if ek != 0:
                    seg.SK_E2.dek += mix_ek
                    seg.SKv3_1.dek += mix_ek
                    seg.Im.dek += mix_ek
                    seg.e_pas += .6*mix_ek

    synapses_list, netstims_list, netcons_list, rnd_list, vecT_list= [], [], [], [], []
    wbin = []


    for i in range(char_arr.shape[0]):
        dend_n = int(char_arr[i,0])
        nmda_syn = h.nmda(m.apic[dend_n](0.99))
        ampa_syn = h.AMPA(m.apic[dend_n](0.99))
        if dek != 0:
            nmda_syn.e = 1

        scalar = 14e-1

        vecStim = h.VecStim()
        stim = StimVec[i,:]
        stim = np.delete(stim, np.where(stim == 0))
        weight = tuning_vm(char_arr[i,-1], 11, stim_alpha)*.7
        if char_arr[i, -1] == 100:
            weight = 0
        if weight != 0:
            wbin.append(weight)

        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_ampa = h.NetCon(vecStim, ampa_syn)
        netCon_ampa.weight[0] = 1*scalar*weight
        netcons_list.append(netCon_ampa)
        synapses_list.append(ampa_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)

        vecStim = h.VecStim()
        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_nmda = h.NetCon(vecStim, nmda_syn)
        netCon_nmda.weight[0] = 1*scalar*weight
        netcons_list.append(netCon_nmda)
        synapses_list.append(nmda_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)


    D_vec = h.Vector().record(m.apic[40](0.0)._ref_v)
    V_vec = h.Vector().record(m.soma(.99)._ref_v)
    t_vec = h.Vector().record(h._ref_t)
    h.finitialize(-50)
    h.tstop = 1500
    h.run()

    #  np.save(f'data/{stim_alpha}.npy', V_vec)
    np.save(f'data_ek/{ek}.npy', V_vec)

#  angles = [0,5,10,15,20,25,30,35,40,45]
dek = np.linspace(0,10,10)

for ek in dek:
    neuron(ek)

#  for a in angles:
#      print(a)
#      neuron(a)


#  plt.plot(t_vec, V_vec)
#  plt.plot(t_vec, D_vec)
#  plt.xlim(1, 400)
#  plt.show()
