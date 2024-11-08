The program is made up of three scripts:

- Create_segments.py:
Functions that can create neuron segements: Soma, Trunk and Dendrites.
The relevant ionic channels are inserted and their currents set according to the Shai model L5PCbiopysics3 with some slight modifications

- Frac_prep.py:
Creates locations for synapses, and assigns synaptic orientations. It further computes the weight given the stimulus orientation. The stimulus orientation is provided as a argument when calling the script. 

- Fractal_neuron.py
Main program, used to simulate the abstract neuron. It simulates the neuron with similar distributed input in the apical dendrites at different orientations. The simulation is ran multiple times to generate many different versions of the neuron. 

- Trunk_length_test.py
Similar to the fractal_neuron.py, main difference being that stimulus orientation is always at target preferred orientation, and instead we simulate the neuron with different neuron sizes
