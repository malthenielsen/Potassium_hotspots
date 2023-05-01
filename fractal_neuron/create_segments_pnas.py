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
    #  dend.insert('Ih')
    dend.ek = -85
    dend.ena = 50
    dend.cm = 2
    dend.g_pas = .6e-4
    dend.decay_CaDynamics_E2 = 35.725651
    dend.gamma_CaDynamics_E2 = 0.000637
    dend.gSK_E2bar_SK_E2 = 2e-3
    dend.gCa_HVAbar_Ca_HVA = 0.05e-3
    dend.gCa_LVAstbar_Ca_LVAst = 0.15e-3 
    dend.gSKv3_1bar_SKv3_1 = 1e-3
    dend.gNaTs2_tbar_NaTs2_t = 5e-3
    dend.gImbar_Im = 0.1e-3
    #  dend.gIhbar_Ih =  1.5e-4
    dend.e_pas = -70
    diam = define_diam(L)
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
    trunk.gSK_E2bar_SK_E2 = 2e-3
    trunk.gCa_HVAbar_Ca_HVA = 0.05e-3
    trunk.gCa_LVAstbar_Ca_LVAst = 0.15e-3 
    trunk.gSKv3_1bar_SKv3_1 = 1e-3
    trunk.gNaTs2_tbar_NaTs2_t = 5e-3
    trunk.gImbar_Im = 0.1e-3
    #  dend.gIhbar_Ih =  1.5e-4
    trunk.e_pas = -70
    #  diam = define_diam(L)
    trunk.diam = 4
    #  trunk.gIhbar_Ih =  .00001
    trunk.e_pas = -70
    return trunk

def create_soma(name):
    soma = h.Section(name = name)
    soma.nseg = 11
    soma.Ra = 100
    soma.insert('Im') 
    soma.insert('Ca_LVAst') 
    soma.insert('Ca_HVA')
    soma.insert('CaDynamics_E2') 
    soma.insert('SK_E2')
    soma.insert('SKv3_1')
    soma.insert('NaTs2_t')
    soma.ek = -80
    soma.ena = 50
    soma.insert('Ih')
    soma.insert('pas')
    soma.gIhbar_Ih = 0.0001*0.75e-3
    soma.g_pas = .3e-5
    soma.gImbar_Im = 0.22e-3
    soma.decay_CaDynamics_E2 = 294.679571#*2/3 
    soma.gamma_CaDynamics_E2 = 0.000509 
    soma.gCa_LVAstbar_Ca_LVAst = 0.01e-3
    soma.gCa_HVAbar_Ca_HVA = 0.9e-3
    soma.gSK_E2bar_SK_E2 = 500e-3
    soma.gSKv3_1bar_SKv3_1 = 100e-3
    soma.gNaTs2_tbar_NaTs2_t = 80e-3 
    soma.e_pas = -70
    soma.cm = 1
    soma.diam = 30
    soma.L = 30
    return soma

