import numpy as np
from matplotlib import pyplot as plt
#  plt.style.use('science')
from neuron import h
#  h.load_file('stdrun.hoc')
#  h.nrn_load_dll('./mod_shai/x86_64/libnrnmech.so')
from scipy import stats


def define_diam(L):
    if L == 20:
        diam = .6
    elif L == 40:
        diam = .8
    elif L == 80:
        diam = 1
    else:
        diam = 1.2
    return diam


def create_dend(name, L):
    dend = h.Section(name = name)
    dend.L = L
    #  dend.nseg = L + 1
    dend.nseg = 11
    #  dend.Ra = find_r(L)
    dend.Ra = 100
    dend.insert('pas')
    dend.insert('CaDynamics_E2')
    dend.insert('SK_E2')
    dend.insert('Ca_LVAst')
    dend.insert('Ca_HVA')
    dend.insert('SKv3_1')
    dend.insert('NaTs2_t')
    dend.insert('Im')
    dend.insert('Ih')
    dend.ek = -85
    dend.ena = 50
    dend.cm = 1
    dend.g_pas = .6e-4
    dend.decay_CaDynamics_E2 = 35.725651
    dend.gamma_CaDynamics_E2 = 0.000637
    dend.gSK_E2bar_SK_E2 = .02e-4
    dend.gCa_HVAbar_Ca_HVA = 7.01e-4 
    dend.gCa_LVAstbar_Ca_LVAst = 22.7e-4 
    dend.gSKv3_1bar_SKv3_1 = 18.08e-4
    dend.gNaTs2_tbar_NaTs2_t = 0.021489*.01
    dend.gImbar_Im = 9.9e-4
    dend.gIhbar_Ih =  1.5e-4
    dend.e_pas = -70
    #dend.diam = define_diam(L)
    return dend

def create_trunk(name, L):
    trunk = h.Section(name = name)
    trunk.nseg = 11
    trunk.Ra = 200
    trunk.insert('pas')
    trunk.insert('CaDynamics_E2')
    trunk.insert('SK_E2')
    trunk.insert('Ca_LVAst')
    trunk.insert('Ca_HVA')
    trunk.insert('SKv3_1')
    trunk.insert('NaTs2_t')
    trunk.insert('Im')
    trunk.insert('Ih')
    trunk.ek = -85
    trunk.ena = 50
    trunk.cm = 2
    trunk.g_pas = .6e-4
    trunk.decay_CaDynamics_E2 = 35.725651
    trunk.gamma_CaDynamics_E2 = 0.000637
    trunk.gSK_E2bar_SK_E2 = .02e-4
    trunk.gCa_HVAbar_Ca_HVA = 7.01e-4 
    trunk.gCa_LVAstbar_Ca_LVAst = 22.7e-4 
    trunk.gSKv3_1bar_SKv3_1 = 18.08e-4
    trunk.gNaTs2_tbar_NaTs2_t =1*0.021489
    trunk.gImbar_Im = 9.9e-4
    trunk.gIhbar_Ih =  1.5e-4
    trunk.e_pas = -70
    #  diam = define_diam(L)
    trunk.diam = 4
    trunk.gIhbar_Ih =  .5e-4
    trunk.e_pas = -70
    return trunk

def create_soma(name):
    soma = h.Section(name = name)
    soma.nseg = 11
    soma.Ra = 100
    soma.insert('Im')
    soma.insert('Ih')
    soma.insert('K_Tst')
    soma.insert('Nap_Et2')
    soma.insert('NaTa_t')
    soma.insert('SK_E2')
    soma.insert('SKv3_1')
    #  soma.insert('NaTs2_t')
    soma.insert('CaDynamics_E2')
    soma.insert('K_Pst')
    soma.insert('Ca_LVAst')
    soma.ek = -85
    soma.ena = 50
    soma.insert('pas')
    soma.g_pas = .3e-5
    #  soma.gImbar_Im = 1.5e-3
    soma.gIhbar_Ih = 0.0002
    soma.decay_CaDynamics_E2 = 460.725651
    soma.gamma_CaDynamics_E2 = 0.000501
    soma.gK_Tstbar_K_Tst = 0.0812
    soma.gK_Pstbar_K_Pst = 0.0023
    soma.gCa_LVAstbar_Ca_LVAst = 0.000343
    soma.gSK_E2bar_SK_E2 = 0.0441
    soma.gSKv3_1bar_SKv3_1 = 0.22
    #  soma.vshift_SKv3_1 = -2
    #  soma.vshift_SK_E2 = -2
    #  soma.gNaTs2_tbar_NaTs2_t = 3*2.04
    soma.gNap_Et2bar_Nap_Et2 = 0*0.00172
    soma.gNaTa_tbar_NaTa_t = 0*2.04
    #soma.vshift_NaTs2_t = -0 

    soma.e_pas = -90
    soma.cm = 1
    soma.diam = 10
    soma.L = 30
    return soma

def goodonecreate_soma(name):
    soma = h.Section(name = name)
    soma.nseg = 11
    soma.Ra = 100
    soma.insert('Im')
    soma.insert('Ih')
    soma.insert('K_Tst')
    soma.insert('Nap_Et2')
    soma.insert('NaTa_t')
    soma.insert('SK_E2')
    soma.insert('SKv3_1')
    #  soma.insert('NaTs2_t')
    soma.insert('CaDynamics_E2')
    soma.insert('K_Pst')
    soma.insert('Ca_LVAst')
    soma.ek = -85
    soma.ena = 50
    soma.insert('pas')
    soma.g_pas = .3e-5
    #  soma.gImbar_Im = 1.5e-3
    soma.gIhbar_Ih = 0.0002
    soma.decay_CaDynamics_E2 = 460.725651
    soma.gamma_CaDynamics_E2 = 0.000501
    soma.gK_Tstbar_K_Tst = 0.0812
    soma.gK_Pstbar_K_Pst = 0.0023
    soma.gCa_LVAstbar_Ca_LVAst = 0.000343
    soma.gSK_E2bar_SK_E2 = 0.0441
    soma.gSKv3_1bar_SKv3_1 = 0.22
    #  soma.vshift_SKv3_1 = -2
    #  soma.vshift_SK_E2 = -2
    #  soma.gNaTs2_tbar_NaTs2_t = 3*2.04
    soma.gNap_Et2bar_Nap_Et2 = 0.00172
    soma.gNaTa_tbar_NaTa_t = 2.04
    #soma.vshift_NaTs2_t = -0 

    soma.e_pas = -90
    soma.cm = 1
    soma.diam = 10
    soma.L = 30
    return soma
