# Local changes in potassium ions regulate input integration in active dendrites


Figure 1 was primarly generated using the folder "ionic_change". 
* Ptest.py generates samples of dendritic segments and computes the factor curve
* Illustration.py creates the majority of the figure
* Line_plots.py makee the remaining plots

Figure 2 was centered around the point dendrite model. To utilize the point dendrite model for other project,
one can use the entire model by lacting point_dendrite.py and Nassi_channels.py in the same subfolder. 
Figures was created by utilizing the point dendrite model in different ways

Figure 3. Channels and IV curves was generated using the point dendrite model. 
Figure 3 was created using fixedpoint.py and the illustration for the Supplementary movie was created usiing nmda_gif.py

Figure 4. Point dendrite was created as hardcoded neuron model inside of each model. 
* Trunk test data and files kan be seen in the subfolder trunk_test
* Tuning curve scripts and fits can be found the subfolder angles_data
* mod files for neuron can be found in the folder mod
