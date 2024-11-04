import numpy as np
from matplotlib import pyplot as plt
plt.style.use('K_PAPER')
from neuron import h, gui

h.load_file('import3d.hoc')

class MyCell:
    def __init__(self):
        morph_reader = h.Import3d_Neurolucida3()
        morph_reader.input('/home/nordentoft/Documents/Potassium_and_dendrites/supplementary_model/morphologies/cell3.asc')
        i3d = h.Import3d_GUI(morph_reader, 0)
        i3d.instantiate(self)

m = MyCell()

print(dir(m))
print(dir(m.apic[3]))

print('%d apic sections' % len(m.apic))
sp = h.PlotShape()
sp.show(0)  # show diameters
