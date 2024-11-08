The program is made up of three scripts:

- Create_segments.py:
Functions that can create neuron segements: Soma, Trunk and Dendrites.
In here the relevant ionic channels is inserted and their currents set according
to the Shai model L5PCbiopysics3 with some slight modifications.

- Frac_prep.py:
Creates locations for synapses when running the script, aswell as assigning it a
orientation and computes the weight in regards to the orientaion we are
stimulating at. This stimulation orientation is given as a argument when calling
the script. 

- Fractal_neuron.py
Main program, used to simulate the simple neuron. If ran, it will create
simulate the neuron with similar distributed input in the apical dendrites at
different orientations. The test is ran multiple times to generate many
different versions of the neuron. To plot the data and analyse the result, use
the tuning scripts found in the folder "Angles_cluster" 

- Trunk_length_test.py
Similar to the fractal_neuron.py, main difference that orientation is always at
target preffered orientation, and instead we test the neuron with different
trunk lengths
