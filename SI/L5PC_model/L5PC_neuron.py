import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('K_PAPER')
from neuron import h, gui
h.nrn_load_dll('./mod_shai/x86_64/libnrnmech.so')
#h.nrn_load_dll('/home/nordentoft/Documents/Potassium_and_dendrites/fractal_neuron/mod_shai/x86_64/libnrnmech.so')
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



h.load_file('import3d.hoc')



class MyCell:
    def __init__(self):
        morph_reader = h.Import3d_Neurolucida3()
        #  morph_reader.input('/home/nordentoft/Documents/Potassium_and_dendrites/supplementary_model/morphologies/cell3.asc')
        morph_reader.input('cell3.asc')
        i3d = h.Import3d_GUI(morph_reader, 0)
        i3d.instantiate(self)

def create_neuron(ek, mix_ek, stim_alpha, not_keep, ID):
    weight_alt = mix_ek
#  def nn(weight_alt):
    ek = 0
    m = MyCell()
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
        apic.insert('Ih')
        apic.insert('pas')
        apic.ek =-85
        apic.ena = 50
        apic.cm = 1
        apic.g_pas = 6e-4
        apic.decay_CaDynamics_E2 = 35.725651
        apic.gamma_CaDynamics_E2 = 0.000637
        apic.gSK_E2bar_SK_E2 = 0.02e-4
        apic.gCa_HVAbar_Ca_HVA = 0.000701
        apic.gCa_LVAstbar_Ca_LVAst = 22.7e-4 
        apic.gSKv3_1bar_SKv3_1 = 0.001808
        apic.gNaTs2_tbar_NaTs2_t = 0.021489*0.01
        apic.gImbar_Im = 0.00099
        apic.gIhbar_Ih = 1.5e-4
        apic.e_pas = -70
        #  apic.gIhbar_Ih =  .00015 //0.00001*1.5

    for al in m.all:
        al.insert('pas')
        #  al.cm = 2
        al.Ra = 100
        #  al.e_pas = -90
        al.nseg =4

    for axon in m.axon:
        axon.insert('pas')
        axon.insert('Im')
        axon.insert('Ca_LVAst') 
        axon.insert('Ca_HVA')
        axon.insert('CaDynamics_E2') 
        axon.insert('SKv3_1')
        axon.insert('SK_E2')
        axon.insert('K_Tst')
        axon.insert('K_Pst')
        axon.insert('Nap_Et2') 
        axon.insert('NaTa_t')
        #  axon.insert('Ih')
        axon.ek = -95
        axon.ena = 50
        axon.g_pas = 3e-5
        #  axon.gIhbar_Ih = 0.0001/2
        axon.gImbar_Im = 0.013322
        axon.decay_CaDynamics_E2 = 277.300774 
        axon.gamma_CaDynamics_E2 = 0.000525 
        axon.gCa_LVAstbar_Ca_LVAst = 0.000813 
        axon.gCa_HVAbar_Ca_HVA = 0.000222 
        axon.gSKv3_1bar_SKv3_1 = 0.473799
        axon.gSK_E2bar_SK_E2 = 0.000047
        axon.gK_Tstbar_K_Tst = 0.077274
        axon.gK_Pstbar_K_Pst = 0.188851
        axon.gNap_Et2bar_Nap_Et2 = 0.005834 
        axon.gNaTa_tbar_NaTa_t = 3.89618

    for soma in m.soma:
        soma.insert('Im')
        soma.insert('Ih')
        #  soma.insert('K_Tst')
        soma.insert('Ca_LVAst')
        soma.insert('Ca_HVA')
        soma.insert('CaDynamics_E2')
        soma.insert('SK_E2')
        soma.insert('SKv3_1')
        soma.insert('NaTs2_t')
        #  soma.insert('Nap_Et2')
        soma.insert('pas')
        soma.ek = -85
        soma.ena = 50
        soma.g_pas = 3e-5
        soma.gIhbar_Ih = 0.0002
        #  soma.gImbar_Im = 0.000008*0.1
        soma.gIhbar_Ih = 0.0002
        soma.decay_CaDynamics_E2 = 460.725651
        soma.gamma_CaDynamics_E2 = 0.000501
        #  soma.gK_Tstbar_K_Tst = 0.0812
        #  soma.gK_Pstbar_K_Pst = 0.0023
        soma.gCa_LVAstbar_Ca_LVAst = 0.000343
        soma.gSK_E2bar_SK_E2 = 0.0441
        soma.gSKv3_1bar_SKv3_1 = 0.22
        soma.gNaTs2_tbar_NaTs2_t = 2.04
        #  soma.gNap_Et2bar_Nap_Et2 = 0.00172
        #  soma.gNaTa_tbar_NaTa_t = 2.04
        soma.e_pas = -90
        soma.cm = 1


    #not_keep = 0

    #stim_alpha = 10
    test = 'none'
    #ID = 0
    dek = 1
    #  ek =10
    #mix_ek = 2

    if not_keep:
        print('made new')
        subprocess.call(f"python3 supp_prep_test.py {stim_alpha} {test} {ID}", shell=True)

    char_arr = np.load(f'bin/3CharFrac_{ID}.npy') 
    StimVec = np.load(f'bin/3VecFrac_{ID}.npy')

    idx = np.where(char_arr[:,2] != 0)
    char_arr = char_arr[idx[0],:]
    StimVec = StimVec[idx[0],:]
   #for apic in m.apic:
   #    idx = 10
   #    id_list = np.arange(0,idx, 1)
   #    np.random.shuffle(id_list)
   #    cluster = int(2*idx//4)
   #    mixed = 0
   #    idx_cluster = id_list[:cluster]
   #    for index, seg in enumerate(apic):
   #        if index in idx_cluster:
   #            if ek != 0:
   #                seg.SK_E2.dek += 1*ek
   #                seg.SKv3_1.dek += 1*ek
   #                seg.Im.dek += 1*ek
   #                seg.e_pas += .6*ek
   #        else:
   #            if ek != 0:
   #                seg.SK_E2.dek += 1*mix_ek
   #                seg.SKv3_1.dek += 1*mix_ek
   #                seg.Im.dek += 1*mix_ek
   #                seg.e_pas += .6*mix_ek


    synapses_list, netstims_list, netcons_list, rnd_list, vecT_list= [], [], [], [], []
    wbin = []


    for i in range(char_arr.shape[0]):
        dend_n = int(char_arr[i,0])
        nmda_syn = h.nmda(m.apic[dend_n](0.99))
        ampa_syn = h.AMPA(m.apic[dend_n](0.99))
        if dek != 0:
            nmda_syn.e = 1

        scalar = 30e-1*weight_alt

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
        netCon_ampa.weight[0] = scalar*weight
        netcons_list.append(netCon_ampa)
        synapses_list.append(ampa_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)

        vecStim = h.VecStim()
        vec = h.Vector(np.sort(stim.astype(int)))
        netCon_nmda = h.NetCon(vecStim, nmda_syn)
        netCon_nmda.weight[0] = scalar*weight
        netcons_list.append(netCon_nmda)
        synapses_list.append(nmda_syn)
        vecT_list.append(vecStim)
        vecStim.play(vec)


    D_vec = h.Vector().record(m.apic[40](0.0)._ref_v)
    V_vec = h.Vector().record(m.soma[0](.99)._ref_v)
    t_vec = h.Vector().record(h._ref_t)
    h.finitialize(-50)
    h.tstop = 1200
    h.run()
    #  plt.plot(t_vec, V_vec)
    #  plt.plot(t_vec, D_vec)
    #  plt.xlim(1, 400)
    #  plt.show()
    #  np.save(f'data_weight/{ek}.npy', V_vec)
    return V_vec

#  def run(i):
#      dek = np.linspace(0,10,10)
#      nn(dek[i])

#  for ek in dek:
    #  nn(ek)
#  if __name__ == '__main__':
#      index = np.arange(0,10,1)
#      pool = Pool(4)
#      pool.map(run, index)
#      pool.close()
#      pool.join()

#  dek = [0,6, 10]
#  dek_r = [0,1.6, 5.9]
#  P_dek = np.load('P_bin1.npy')
#  P_mix = np.load('P_bin_off_300.npy')
#  P_mix = P_mix/P_dek.max()
#  P_dek = P_dek/np.max(P_dek)
#  dek_a = interpolate.interp1d(np.linspace(0,90,45), np.nan_to_num(P_dek, 0))
#
def run(i):
    angles = np.linspace(0,35,18)
    angles = np.append(angles, np.array([40, 50, 75, 90]))
    dek_a = np.zeros_like(angles)
    time.sleep(i)
    ID = int(time.time())
    soma_final = []
    dek_r = [0.95, 1, 1.05]
    for index, a in enumerate(angles):
        for idx, ek in enumerate(dek_r):
            if a == 0 and ek == 0.95:
                remix = True
            else:
                remix = 0
            V_vec = create_neuron(dek_a[0], dek_r[idx], a, remix, ID)
            V_vec = np.array(V_vec)
            soma_final.append(V_vec[::100])
            #  break
        #  break

    np.save(f'./dataL5PC_weight/soma_{i}', np.vstack(soma_final))
    #  np.save(f'./dataL5PC_weight/soma_{i}')

#  run(0)
if __name__ == '__main__':
    index = np.arange(0,15,1)
    pool = Pool(15)
    pool.map(run, index)
    pool.close()
    pool.join()

#  plt.plot(t_vec, V_vec)
#  plt.plot(t_vec, D_vec)
#  plt.xlim(1, 400)
#  plt.show()
